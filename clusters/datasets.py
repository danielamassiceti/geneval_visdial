import os, torch, inflect, json, sys, re
from nltk import word_tokenize
from PIL import Image
from torch.utils.data.dataset import Dataset

from corpus import Dictionary
from utils import send_to_device

class VisualDialogDataset(Dataset):
    def __init__(self, mode, args, with_options=False, transform=None):

        self.mode = mode
        self.D = args.D
        self.S = args.S
        self.with_options = with_options
        self.with_human_scores = True
        self.device = args.gpu
        self.transform = transform
        self.num_converter = inflect.engine()
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        # path to visdial .json file for mode 
        self.data_file = '{}/visdial_{}_{}.json'.format(args.dataset_path, str(args.version), mode)
        # path to file with human relevance scores
        self.human_scores_path = '{}/visdial_{}_{}_dense_annotations.json'.format(args.dataset_path, str(args.version), mode)
        # path to directory with saved pre-processed data
        self.preprocessed_data_path = os.path.join(args.dataset_path, '{:}_processed_S{:}_D{:}_wo{:}_whs{:}'.format(mode, self.S, self.D, self.with_options,
            self.with_human_scores))
        # path to file with saved dictionary
        self.saved_dictionary_path = os.path.join(args.dataset_path, 'vocab_visdial_v{:}_train.pt'.format(args.version))
        # path to data file to build dictionary
        self.dictionary_data_path = os.path.join(args.dataset_path, 'visdial_{:}_train.json'.format(args.version))
        # path to .bin FastText model
        self.fast_text_model = args.fast_text_model
        # path to file with saved word vector embeddings
        self.saved_wordvectors_path = os.path.join(args.dataset_path, '{:}_vocab_vecs.pt'.format(os.path.basename(args.fast_text_model)))
        
    def load_data(self):
        
        print(self.mode + ' data loading...')
        
        with open(self.data_file) as f:
            json_data = json.load(f)
        
        self.all_questions = json_data['data']['questions']
        self.all_answers = json_data['data']['answers']

        if os.path.exists(self.preprocessed_data_path):
            self.image_ids = torch.load(os.path.join(self.preprocessed_data_path, 'image_ids.pt'))
            self.questions = torch.load(os.path.join(self.preprocessed_data_path, 'questions.pt'))
            self.question_ids = torch.load(os.path.join(self.preprocessed_data_path, 'question_ids.pt'))
            self.question_lengths = torch.load(os.path.join(self.preprocessed_data_path, 'question_lengths.pt'))
            self.answers = torch.load(os.path.join(self.preprocessed_data_path, 'answers.pt'))
            self.answer_ids = torch.load(os.path.join(self.preprocessed_data_path, 'answer_ids.pt'))
            self.answer_lengths = torch.load(os.path.join(self.preprocessed_data_path, 'answer_lengths.pt'))
            self.in_human_set = torch.load(os.path.join(self.preprocessed_data_path, 'in_human_set.pt'))
            if self.with_options:
                self.gtidxs = torch.load(os.path.join(self.preprocessed_data_path, 'gtidxs.pt'))
                self.answer_options = torch.load(os.path.join(self.preprocessed_data_path, 'answer_options.pt'))
                self.answer_options_ids = torch.load(os.path.join(self.preprocessed_data_path, 'answer_options_ids.pt'))
                self.answer_options_lengths = torch.load(os.path.join(self.preprocessed_data_path, 'answer_options_lengths.pt')) 
                if self.with_human_scores:
                    self.answer_options_scores = torch.load(os.path.join(self.preprocessed_data_path, 'answer_options_scores.pt'))
            
            self.N = len(self.image_ids)
            self.human_N = torch.sum(self.in_human_set)
        else:
            self.preprocess_data(json_data)

        print(self.mode + ' data loaded: items: {:d}'.format(self.N))

    def preprocess_data(self, json_data):
 
        with open(self.human_scores_path) as hsp:
            hs = json.load(hsp)
        human_scores = {i['image_id']: i for i in hs}

        samples = json_data['data']['dialogs']
        self.N = len(samples)
       
        self.image_ids = []
        self.questions = torch.LongTensor(self.N, self.D).fill_(self.dictionary.UNK_ID)
        self.question_ids = torch.LongTensor(self.N, self.D, self.S).fill_(self.dictionary.UNK_ID)
        self.question_lengths = torch.LongTensor(self.N, self.D).fill_(self.dictionary.UNK_ID)
        self.answers = torch.LongTensor(self.N, self.D).fill_(self.dictionary.UNK_ID)
        self.answer_ids = torch.LongTensor(self.N, self.D, self.S).fill_(self.dictionary.UNK_ID)
        self.answer_lengths = torch.LongTensor(self.N, self.D).fill_(self.dictionary.UNK_ID)
        self.in_human_set = torch.ByteTensor(self.N, self.D).fill_(0)
        if self.with_options:
            self.gtidxs = torch.LongTensor(self.N, self.D).fill_(-1)
            self.answer_options = torch.LongTensor(self.N, self.D, 100).fill_(self.dictionary.UNK_ID)
            self.answer_options_ids = torch.LongTensor(self.N, self.D, 100, self.S).fill_(self.dictionary.UNK_ID)
            self.answer_options_lengths = torch.LongTensor(self.N, self.D, 100).fill_(self.dictionary.UNK_ID)
            if self.with_human_scores:
                self.answer_options_scores = torch.Tensor(self.N, self.D, 100).fill_(0)

        for n, s in enumerate(samples):

            sys.stdout.write('\r{}/{} --> {:3.1f}%'.format(str(n+1), str(self.N), (n+1)/float(self.N)*100))
            sys.stdout.flush()
            self.image_ids.append(str(s['image_id']))
             
            for d, dialog in enumerate(s['dialog']):
                qtokens = word_tokenize(self.preprocess_sentence(self.all_questions[dialog['question']]))
                self.question_ids[n][d], self.question_lengths[n][d] = self.pad_to_max_length(qtokens)
                self.questions[n][d] = dialog['question']
                atokens = word_tokenize(self.preprocess_sentence(self.all_answers[dialog['answer']]))
                self.answer_ids[n][d], self.answer_lengths[n][d] = self.pad_to_max_length(atokens)
                self.answers[n][d] = dialog['answer']
                
                if self.with_options and 'answer_options' in dialog:
                    self.gtidxs[n][d] = dialog['gt_index']
                    for o, option in enumerate(dialog['answer_options']):
                        otokens = word_tokenize(self.preprocess_sentence(self.all_answers[option]))
                        self.answer_options_ids[n][d][o], self.answer_options_lengths[n][d][o] = self.pad_to_max_length(otokens)
                        self.answer_options[n][d][o] = option
            if s['image_id'] in human_scores: # only partial set of train have human relevance scores
                human_item = human_scores[s['image_id']]
                if sum(human_item['gt_relevance']) > 0:
                    self.in_human_set[n][ human_item['round_id'] - 1 ] = 1
                    if self.with_human_scores:
                        self.answer_options_scores[n][ human_item['round_id'] - 1 ] = torch.Tensor(human_item['gt_relevance'])

        sys.stdout.write("\n")
        self.human_N = torch.sum(self.in_human_set)
        
        # save to file
        os.makedirs(self.preprocessed_data_path, exist_ok=True)
        torch.save(self.image_ids, os.path.join(self.preprocessed_data_path, 'image_ids.pt'))
        torch.save(self.questions, os.path.join(self.preprocessed_data_path, 'questions.pt'))
        torch.save(self.question_ids, os.path.join(self.preprocessed_data_path, 'question_ids.pt'))
        torch.save(self.question_lengths, os.path.join(self.preprocessed_data_path, 'question_lengths.pt'))
        torch.save(self.answers, os.path.join(self.preprocessed_data_path, 'answers.pt'))
        torch.save(self.answer_ids, os.path.join(self.preprocessed_data_path, 'answer_ids.pt'))
        torch.save(self.answer_lengths, os.path.join(self.preprocessed_data_path, 'answer_lengths.pt'))
        torch.save(self.in_human_set, os.path.join(self.preprocessed_data_path, 'in_human_set.pt'))
        if self.with_options:
            torch.save(self.gtidxs, os.path.join(self.preprocessed_data_path, 'gtidxs.pt'))
            torch.save(self.answer_options, os.path.join(self.preprocessed_data_path, 'answer_options.pt'))
            torch.save(self.answer_options_ids, os.path.join(self.preprocessed_data_path, 'answer_options_ids.pt'))
            torch.save(self.answer_options_lengths, os.path.join(self.preprocessed_data_path, 'answer_options_lengths.pt')) 
            if self.with_human_scores:
                torch.save(self.answer_options_scores, os.path.join(self.preprocessed_data_path, 'answer_options_scores.pt'))
    
    def load_dictionary(self, shared_dictionary=None):
        
        if shared_dictionary:
            self.dictionary = shared_dictionary
        else:
            print('dictionary and word vectors loading...')
            if os.path.exists(self.saved_dictionary_path):
                self.dictionary = torch.load(self.saved_dictionary_path)
            else:
                self.build_dictionary()
            self.dictionary.load_word_vectors(self.fast_text_model, self.saved_wordvectors_path)
            self.dictionary.word_vectors = send_to_device(self.dictionary.word_vectors, self.device)

            print('dictionary loaded: words: {:d}'.format(len(self.dictionary)))
            print('word vectors loaded: words: {:d}; {:d}-dim'.format(len(self.dictionary.word_vectors), self.dictionary.emb_size))


    def build_dictionary(self):
        
        self.dictionary = Dictionary()

        special_tokens = [self.pad_token, self.eos_token, self.unk_token]
        for t in special_tokens:
            self.dictionary.add_word(t)
        self.dictionary.set_UNK_ID(self.unk_token)
    
        with open(self.dictionary_data_file) as data_file:
            jdata = json.load(data_file)

        questions = jdata['data']['questions']
        answers = jdata['data']['answers']

        for i in jdata['data']['dialogs']:

            dialog = i['dialog'] #get dialog for that image
        
            ctokens = word_tokenize(self.preprocess_sentence(i['caption'])) 
            for ct in ctokens:
                self.dictionary.add_word(ct)
        
            ndialog = len(dialog)
            for d in dialog:
                candidates = d['answer_options']
        
        	# preprocess & tokenize
                q_idx = d['question']
                a_idx = d['answer']
                qtokens = word_tokenize(self.preprocess_sentence(questions[q_idx]))
                atokens = word_tokenize(self.preprocess_sentence(answers[a_idx]))
            
                allotokens = []
                for t in qtokens+atokens+allotokens:
                    self.dictionary.add_word(t)

        self.dictionary.filterbywordfreq(special_tokens, 5) # remove words with freq of < 5 

    def __getitem__(self, n):

        sample = {
                   'image_ids' : self.image_ids[n],
                   'questions' : self.questions[n],
                   'questions_ids' : self.question_ids[n], 
                   'questions_length' : self.question_lengths[n],
                   'answers' : self.answers[n],
                   'answers_ids' : self.answer_ids[n],
                   'answers_length' : self.answer_lengths[n],
                   'in_human_set' : self.in_human_set[n]
                 }

        if self.with_options:
            sample['gtidxs'] = self.gtidxs[n]
            sample['answer_options'] = self.answer_options[n]
            sample['answer_options_ids'] = self.answer_options_ids[n]
            sample['answer_options_length'] = self.answer_options_lengths[n]
            if self.with_human_scores:
                sample['answer_options_scores'] = self.answer_options_scores[n]
        
        return sample

    def __len__(self):
        return self.N

    # gets token list to be of length opt.seqlen using EOS and PAD tokens
    # if questions/answers are shorter than opt.seqlen, then pad with PAD
    # if questions/answers are longer than opt.seqlen, then chop-chop!
    def pad_to_max_length(self, tokenlist):

        PAD_ID = self.dictionary.word2idx[self.pad_token]
        EOS_ID = self.dictionary.word2idx[self.eos_token]
        UNK_ID = self.dictionary.word2idx[self.unk_token]

        tokenlist = [self.dictionary.word2idx.get(t, UNK_ID) for t in tokenlist]
        T = len(tokenlist)
    
        suffix = [EOS_ID]
        if T < self.S:
            suffix.extend( [ PAD_ID for x in range(self.S - T - 1) ] ) # add EOS PAD PAD PAD ...
            return torch.LongTensor(tokenlist+suffix), T+1
        else:
            return torch.LongTensor(tokenlist[0:self.S-1]+suffix), self.S

    def preprocess_sentence(self, sentence):

        # remove all apostrophes
        sentence = sentence.replace("'", "")
        sentence = re.sub(r'\d+', lambda x: self.digitreplacer(x.group()), sentence).lower()
    
        return sentence
    
    # convert digits to words
    def digitreplacer(self, digit):
        return self.num_converter.number_to_words((digit)).replace("-"," ")

    def get_avg_embedding(self, sentences, lengths):

        b,d,s= sentences.size()
        word_embeddings = self.dictionary.word_vectors.index_select(0, sentences.view(-1)).view(b,d,s,-1)
        return word_embeddings.sum(dim=2).div_(lengths.float().unsqueeze(2))


 
