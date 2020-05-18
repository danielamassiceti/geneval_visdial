import os, torch
from gensim.models import FastText

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.freq = {}
 
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.freq[word] = 1
        else:
            self.freq[word] = self.freq[word] + 1
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)

    def set_UNK_ID(self, UNK_ID):
        self.UNK_ID = self.word2idx[UNK_ID]

    def word_vectors_to_device(device):
        self.word_vectors = self.word_vectors.to('cuda:' + str(device))

    def load_word_vectors(self, word_emb_model, saved_wordvectors_path):

        self.emb_size = 300
        if os.path.exists(saved_wordvectors_path):
            self.word_vectors = torch.load(saved_wordvectors_path)
        else:
            self.build_word_vectors(word_emb_model)

    def build_word_vectors(self, word_emb_model):
        
        wvs = FastText.load_fasttext_format(word_emb_model)
        self.emb_size = 300
            
        word_vectors, null_counter = [], 0
        for i, idx in enumerate(self.idx2word):
            try:
                word_vectors.append(torch.from_numpy(wvs[idx]).float())
            except KeyError:
                null_counter+=1
                word_vectors.append(torch.zeros(self.emb_size))
           
        self.word_vectors = torch.stack(word_vectors)
        self.word_vectors[0].fill_(0) # fill embedding for <PAD> with 0s
        torch.save(self.word_vectors, self.saved_wordvectors_path)
        print('number of words without a pre-trained word embedding: {:d}/{:d}'.format(null_counter, len(self.idx2word)))
            
    def filterbywordfreq(self, special_tokens, n_freq=5):
        new_word2idx = {}
        new_idx2word = []
        new_freq = {}
        for idx in range(len(self.word2idx)):
            word = self.idx2word[idx]
            if self.freq[word] >= n_freq or word in special_tokens:
                new_idx2word.append(word)
                new_word2idx[word] = len(new_idx2word)-1
                new_freq[word] = self.freq[word]
        self.idx2word = new_idx2word
        self.word2idx = new_word2idx
        self.freq = new_freq

