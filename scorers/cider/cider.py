# Adapted from https://github.com/vrama91/cider/blob/master/pyciderevalcap/cider/cider_scorer.py
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy, pickle, math, os
from collections import defaultdict
import numpy as np

def precook(s, n=4):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = tuple(words[i:i+k])
            counts[ngram] += 1
    return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def cook_test(test, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    '''
    return precook(test, n)

class CiderScorer(object):
    """CIDEr scorer.
    """

    def __init__(self, refs, n=4, sigma=6.0, df_mode="corpus"):
        ''' singular instance '''
        self.n = n
        self.sigma = sigma
        self.document_frequency = defaultdict(int)
        self.df_mode = df_mode
        
        # compute idf
        if self.df_mode == "corpus":
            self.compute_doc_freq(refs)
        else:
            self.document_frequency = pickle.load(open(os.path.join('data', df_mode + '.p'),'r'))

    def compute_doc_freq(self, crefs):
        '''
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        '''
        for refs in crefs:
            cooked_refs = cook_refs(refs, n=self.n)
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in cooked_refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
        
        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(crefs)))
        elif self.df_mode == "coco-val-df":
            # if coco option selected, use length of coco-val set
            self.ref_len = np.log(float(40504)) 

    def compute_cider(self, hypothesis, references):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram,term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram)-1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                # compute norm for the vector.  the norm will be used for
                # computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            '''
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            '''
            delta = float(length_hyp - length_ref)
            # measure cosine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram,count) in vec_hyp[n].items():
                    val[n] += vec_hyp[n][ngram] * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])

                assert(not math.isnan(val[n]))
            return val

        # compute vector for test captions
        cooked_test = cook_test(hypothesis, n=self.n)
        vec, norm, length = counts2vec(cooked_test)
        # compute vector for ref captions
        cooked_refs = cook_refs(references, n=self.n)
        score = np.array([0.0 for _ in range(self.n)])
        for cooked_ref in cooked_refs:
            vec_ref, norm_ref, length_ref = counts2vec(cooked_ref)
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
        # divide by number of references
        score /= len(references)
        # change by vrama91 - mean of ngram scores, instead of sum 
        score_avg=np.array([0.0 for _ in range(self.n)])
        for n in range(self.n):
            score_avg[n] = np.mean(score[:n+1])
        return score_avg.tolist()

    def compute_score(self, hypothesis, references):
        # compute cider score
        return self.compute_cider(hypothesis, references)

