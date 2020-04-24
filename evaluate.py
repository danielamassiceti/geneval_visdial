import os
import re
import sys
import json
import torch
import inflect
import argparse
import numpy as np
from nltk import word_tokenize
from scorers.cider.cider import CiderScorer
from scorers.meteor.meteor import Meteor
import gensim.models.fasttext as FastText
from bert_serving.client import BertClient
from nltk.tokenize.treebank import TreebankWordDetokenizer

def main():

    parser = argparse.ArgumentParser(description='Generative Evaluation for Visual Dialogue')
    parser.add_argument('--generations', dest='generations', default='./generations.json', help='Path to file with answer generations.')
    parser.add_argument('--references', dest='references', default='densevisdial/refs_S_val.json', help='Path to file with answer reference sets.')

    # overlap (CIDER, METEOR) parameters
    parser.add_argument('--n', dest='n', type=int, default=4, help='Cider n-gram (computes 1 to n).')
    parser.add_argument('--no_overlap', dest='no_overlap', action='store_true', help='Do not compute overlap metrics.')
   
    # embedding distance FastText parameters
    parser.add_argument('--fast_text_model', dest='fast_text_model', required=True, help='Path to FastText .bin model.')
    parser.add_argument('--no_embedding', dest='no_embedding', action='store_true', help='Do not compute embedding metrics.')
    args = parser.parse_args()

    # load answer generations and reference sets
    print ('loading generations and references from .json files...')
    with open(args.generations) as f:
        gens = json.load(f)
    with open(args.references) as f:
        refs = json.load(f)

    print ('preparing data...')
    generations, references = prepare_data(gens, refs)

    print ('# question-answer pairs: ' + str(len(refs)))
   
    # load models
    print ('loading models and word embeddings (may take a few minutes)...')
    if not args.no_overlap:
        cider_model = CiderScorer(references, n=args.n)
        meteor_model = Meteor()
    if not args.no_embedding:
        bert_client = BertClient(check_length=False)
        fasttext_wordvectors = FastText.load_facebook_vectors(args.fast_text_model)
        numconverter = inflect.engine()
    print ('models loaded!')

    scores = initialise_score_dicts(args)
    print ('evaluating generations...')
    for i, (gs, rs) in enumerate(zip(generations, references)):
        sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(i+1), str(len(references)), (i+1)/float(len(references))*100))
        sys.stdout.flush()
        
        cider_list, meteor_list = [], []
        bert_list, fasttext_list = [], []
    
        # get bert embeddings of references
        if not args.no_embedding:
            bert_refs = get_bert_features(rs, bert_client)
            fasttext_refs = get_fasttext_features(rs, fasttext_wordvectors, numconverter)
        
        for ii, g in enumerate(gs): # loops through answer generations, if multiple
            
            if g == "": # ignore empty string
                scores['empty'] += 1
            else:

                if not args.no_overlap:
                    cider_list.append(compute_cider(g, rs, cider_model))
                    meteor_list.append(compute_meteor(g, rs, meteor_model))
                if not args.no_embedding:
                    bert_list.append(compute_bert(g, bert_refs, bert_client))
                    fasttext_list.append(compute_fasttext(g, fasttext_refs, fasttext_wordvectors, numconverter))
    
        # average over multiple generations
        if not args.no_overlap:
            n_grams_cider = np.mean(cider_list, axis=0)
            for n, n_gram_cider in enumerate(n_grams_cider):
                scores['cider_{:d}'.format(n+1)].append(n_gram_cider)
            scores['meteor'].append(np.mean(meteor_list))
        if not args.no_embedding:
            bert_scores = np.mean(bert_list, axis=0)
            scores['bert_l2'].append(bert_scores[0])
            scores['bert_cs'].append(bert_scores[1])
            fasttext_scores = np.mean(fasttext_list, axis=0)
            scores['fasttext_l2'].append(fasttext_scores[0])
            scores['fasttext_cs'].append(fasttext_scores[1])
    
    sys.stdout.write('\n')
    print_scores(scores)
    if 'meteor' in scores:
        meteor_model.close()

def initialise_score_dicts(args): 
    
    scores = {}
    # overlap
    if not args.no_overlap:
        for n in range(args.n):
            scores['cider_{:d}'.format(n+1)] = []
        scores['meteor'] = []
    # embedding
    if not args.no_embedding:
        scores['bert_l2'] = []
        scores['bert_cs'] = []
        scores['fasttext_l2'] = []
        scores['fasttext_cs'] = []
    # admin
    scores['empty'] = 0

    return scores

def print_scores(scores):

    headings = ""
    output = ""
    tb = "\t"
    for metric,scores_list in scores.items():
        headings += metric + tb + tb
        output += '{0:.4f} ({1:.4f})'.format(np.mean(scores_list), np.std(scores_list)) + tb

    print ('--'*10)
    print (headings)
    print ('--'*10)
    print (output)
    print ('--'*10)

def prepare_data(gens, refs):
    
    sorted_gens = sorted(gens, key=lambda k: (k['image_id'], k['round_id']))
    sorted_refs = sorted(refs, key=lambda k: (k['image_id'], k['round_id']))
    offset = 1 if (len(gens) == len(refs)) else int(len(sorted_gens)/len(sorted_refs)) 
    dt = TreebankWordDetokenizer()
    
    generations, references = [], []
    for i, refs in enumerate(sorted_refs): 
        sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(i+1), str(len(sorted_refs)), (i+1)/float(len(sorted_refs))*100))
        sys.stdout.flush()
        if offset == 1:
            gens = sorted_gens[i]
        else:
            gens = sorted_gens[i * offset + refs['round_id'] ]

        # ensure gens and refs correspond to same image/round
        assert (gens['image_id'] == refs['image_id'])
        assert (gens['round_id'] == refs['round_id'])
       
        # list of generated answers (can be multiple generated answer per entry)
        generations.append( [dt.detokenize(word_tokenize(a_gen)) for a_gen in gens['generations'] ] )
        
        # list of references answers
        references.append( refs['refs'] )
        #references.append( [dt.detokenize(word_tokenize(a_ref)) for a_ref in refs['refs']] )
    
    sys.stdout.write('\n')  
    return generations, references
   
def compute_cider(generation, references, cider_model):
  
    return cider_model.compute_score(generation, references)
        
def compute_meteor(generation, references, meteor_model):
    
    score, _ = meteor_model.compute_score(generation, references)

    return score

def get_fasttext_features(sentences, wordvectors, numconverter):
    
    if not isinstance(sentences, list):
        sentences = [sentences]
   
    # get fasttext embedding
    emb_sentences = []
    for sentence in sentences:
        clean_sentence = re.sub(r'\d+', lambda x: digitreplacer(x.group(), numconverter), sentence).lower()
        vecs=[]
        words = clean_sentence.split()
        for word in words:
            vecs.append( wordvectors[ word ])
        emb_sentences.append( np.mean(vecs, axis=0) )

    return torch.Tensor( emb_sentences ).view(-1, 300)
     
def compute_fasttext(generation, fasttext_references, wordvectors, numconverter):
 
    # get fasttext avg embedding of generation
    fasttext_generation = get_fasttext_features(generation, wordvectors, numconverter)
     
    # l2 distance
    l2 = torch.nn.PairwiseDistance(p=2)(fasttext_references, fasttext_generation.expand_as(fasttext_references))
    # cosine similarity
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(fasttext_references, fasttext_generation.expand_as(fasttext_references))

    return l2.mean().item(), cosine.mean().item()

def get_bert_features(sentences, bert_client):
    
    if not isinstance(sentences, list):
        sentences = [sentences]
    
    emb_sentences = bert_client.encode(sentences)
    
    return torch.Tensor(emb_sentences).view(-1, 768)

def compute_bert(generation, bert_references, bert_client):
    
    # get bert embedding of generation
    bert_generation = get_bert_features(generation, bert_client)
     
    # l2 distance
    l2 = torch.nn.PairwiseDistance(p=2)(bert_references, bert_generation.expand_as(bert_references))
    # cosine similarity
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)(bert_references, bert_generation.expand_as(bert_references))

    return l2.mean().item(), cosine.mean().item()
       
# convert digits to words
def digitreplacer(digit, numconverter):
    return numconverter.number_to_words((digit)).replace("-"," ")
    
    
    
if __name__ == '__main__':
    main()

