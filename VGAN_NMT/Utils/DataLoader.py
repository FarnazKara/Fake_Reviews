import re
import os
import numpy as np
import math
import torch
from collections import Counter, defaultdict
from Utils.utils import aeq
from Utils.utils import trace
from itertools import chain
import random, pickle
import fixed_data as cfg

PAD_WORD = '<pad>' # 0
UNK_WORD = '<unk>' # 1
BOS_WORD = '<s>'   # 2
EOS_WORD = '</s>'  # 3


class Vocab(object):
    def __init__(self, lang=None, config=None,  **kwargs):
        self.specials = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]
        self.counter = Counter()
        self.stoi = {} 
        self.itos = {}
        self.lang = lang
        self.weights = None
        self.min_freq = config.min_freq
        self.word_vectors = None

    def make_vocab(self, dataset):
        for x in dataset:
            self.counter.update(x)
        
        if self.min_freq > 1:
            self.counter = {w.lower():i for w, i in filter(
                lambda x:x[1] >= self.min_freq, self.counter.items())}
        self.vocab_size = 0
        for w in self.specials:
            w = w.lower()
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        for w in self.counter.keys():
            
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1
        
        self.itos = {i:w for w, i in self.stoi.items()}
        

    
    def load_pretrained_embedding_not_train_data(self, embed_path, embed_dim):
        self.weights = np.zeros((self.vocab_size, int(embed_dim)))
        with open(embed_path, "r", errors="replace", encoding='utf8') as embed_fin:
            for line in embed_fin:
                cols = line.rstrip("\n").split()
                if len(cols) > 1+embed_dim:
                    w = "".join(cols[0:len(cols)-embed_dim+1])
                else:
                    w = cols[0]
                if w in self.stoi:
                    #print(w)
                    #print(len(np.array(cols)))
                    #print("++++++")
                    if len(cols) > 1+embed_dim:
                        weight = np.array(cols[len(cols)-embed_dim+1:])
                    else:
                        weight = np.array(cols[1:])
                    self.weights[self.stoi[w]] = weight
                else:
                    pass
        embed_fin.close()
        print("num words already in pretrained embedding is: %d" % len(self.weights))
        print("vocabulary size: %d" % self.vocab_size)
        #for i in range(1, 2):
            #self.weights[i] = np.zeros((embed_dim,))
            # self.weights[i] = np.random.random_sample(
            #     (self.config.embed_dim,))
        self.weights = torch.from_numpy(self.weights)
        print("embedding vector is loaded")
    
    def load_model(self, file_path):
        with open(file_path, 'rb') as fp:
            pickle_obj = pickle.load(fp)
            #print ('loaded %d vectors from %s' % (len(pickle_obj[1]), file_path))
            stoi = pickle_obj[0]
            vocabs_dict = pickle_obj[1]
            vector_dim = len(list(vocabs_dict.values())[0])
            weights = np.zeros((len(stoi), vector_dim))
            for w in vocabs_dict:
                weights[stoi[w]] = vocabs_dict[w]
        return weights

    def save_model(self, file_path):
        print ('dumping %d vectors into %s' % (len(self.weights), file_path))
        with open(file_path, 'wb') as handle:
            pickle.dump([self.stoi,self.word_vectors], handle, protocol=2)
        
        
        #pickle.dump([self.stoi.keys(), self.word_vectors ], open(file_path, 'w'), protocol=2)

    def load_pretrained_embedding(self, embed_path, embed_dim, embed_vector_path):
        if os.path.exists(embed_vector_path):
            self.weights = self.load_model(embed_vector_path)
        else:    
            word_vectors = dict()
            self.weights = np.zeros((self.vocab_size, int(embed_dim)))
            with open(embed_path, "r", errors="replace", encoding='utf8') as embed_fin:
                for line in embed_fin:
                    cols = line.rstrip("\n").split()
                    if len(cols) > 1+embed_dim:
                        w = "".join(cols[0:len(cols)-embed_dim+1])
                    else:
                        w = cols[0]
                    if w in self.stoi:
                        #print(w)
                        #print(len(np.array(cols)))
                        #print("++++++")
                        if len(cols) > 1+embed_dim:
                            weight = np.array(cols[len(cols)-embed_dim+1:])
                        else:
                            weight = np.array(cols[1:])
                        word_vectors[w.lower()] = weight

                        self.weights[self.stoi[w]] = weight
                    else:
                        pass
            embed_fin.close()
            print("num words already in pretrained embedding is: %d" % len(self.weights))
            print("vocabulary size: %d" % self.vocab_size)
            #for i in range(1, 2):
                #self.weights[i] = np.zeros((embed_dim,))
                # self.weights[i] = np.random.random_sample(
                #     (self.config.embed_dim,))
            self.word_vectors = word_vectors
            self.weights = torch.from_numpy(self.weights)
            self.save_model(embed_vector_path)
        print("embedding vector is loaded")

#    def _add_unknown_words(self, embed_dim):
#        if self.vocab_size == self.weights:
#            print ('No randomized words') 
#            return
#        not_present = 0
#        for idx in range(self.vocab_size):
#            if self.weights[idx] ==  np.zeros(int(embed_dim)):
#                self.weights[idx] = np.random.uniform(-0.25, 0.25, 300)
#                not_present += 1
#        print ('randomized words: %d out of %d' % (not_present, len(self.vocab_size)))

    def __getitem__(self, key):
        return self.weights[key]

    def __len__(self):
        return self.vocab_size

class DataSet(list):
    def __init__(self, *args, config=None, is_train=True, dataset="train"):
        self.config = config
        self.is_train = is_train
        self.src_lang = config.src_lang
        self.trg_lang = config.trg_lang
        self.dataset = dataset
        self.ref_dict = {}
        self.data_path = (
            os.path.join(self.config.data_path, dataset + "." + self.src_lang+".csv"),
            os.path.join(self.config.data_path, dataset + "." + self.trg_lang+".csv")
            )
        super(DataSet, self).__init__(*args)

    
    def clean_str(self,my_string):
        cleaned_text = re.sub(r'<[^<]+?>','',my_string)
        cleaned_text = re.sub(r'[a-z]*[:.]+\S+', '', cleaned_text)
        cleaned_text =" ".join(re.sub("[^A-Za-z-0-9.,']+",' ',cleaned_text).split())
        cleaned_text = cleaned_text.replace(',',' , ')
        cleaned_text = cleaned_text.replace('.',' . ')
        cleaned_text = cleaned_text.replace('  ',' ')
        return cleaned_text.lower()
        
    
    
        
    def read(self):
        #Skip the header   
#        with open(self.data_path[0], "r") as fin_src,\
#             open(self.data_path[1], "r") as fin_trg:
#            header = fin_src.readline()
#            header2 = fin_trg.readline()
#            for line1, line2 in zip(fin_src, fin_trg):
#                src, trg = line1.rstrip("\r\n"), line2.rstrip("\r\n")
#                src, trg = (src.replace(',',' ')).lower(), (trg.replace(',', ' ')).lower()
#                src, trg = self.clean_str(src), self.clean_str(trg)
#                src, trg = src.split(), trg.split()
#                src, trg = src[1:] , trg[1:]
#                if len(src) <= self.config.max_seq_len and \
#                    len(trg) <= self.config.max_seq_len:
#                            sent_src = " ".join(src)
#                            if sent_src not in self.ref_dict :
#                                self.ref_dict[sent_src] = []
#                            self.ref_dict[sent_src].append(trg)
#                
#        fin_src.close()
#        fin_trg.close()
        
        
        with open(self.data_path[0], "r") as fin_src,\
             open(self.data_path[1], "r") as fin_trg:
            header = fin_src.readline()
            header2 = fin_trg.readline()
            for line1, line2 in zip(fin_src, fin_trg):
                src, trg = line1.rstrip("\r\n"), line2.rstrip("\r\n")
                src, trg = (src.replace(',',' ')).lower(), (trg.replace(',', ' ')).lower()
                src, trg = self.clean_str(src), self.clean_str(trg)
                src, trg = src.split(), trg.split()
                src, trg = src[1:] , trg[1:]
                if self.is_train:
                    if len(src) <= self.config.max_seq_len and \
                                    len(trg) <= self.config.max_seq_len:
                                self.append((src, trg))#, self.ref_dict[" ".join(src)]))
                else:
                    self.append((src, trg))#, self.ref_dict[" ".join(src)]))
        fin_src.close()
        fin_trg.close()

    def _numericalize(self, words, stoi):
        return  [1 if x not in stoi else stoi[x] for x in words] 
        

    def numericalize(self, src_w2id, trg_w2id):
        #TODO: I added z vector
        for i, example in enumerate(self):
            #x, y, z = example
            x, y = example
            x = self._numericalize(x, src_w2id)
            y = self._numericalize(y, trg_w2id)
#            new_z = []
#            for j in range(len(z)):
#                new_z.append(self._numericalize(z[j], trg_w2id))
#            self[i] = (x, y, new_z)
            self[i] = (x, y)

        print("It is numericalize")

class DataBatchIterator(object):
    def __init__(self, config, dataset="train", 
                    is_train=True, batch_size=64, 
                    shuffle=False, sample=False, 
                    sort_in_batch=True):
        self.config = config
        self.examples = DataSet(config=config, is_train=is_train, dataset=dataset)
        self.src_vocab = Vocab(lang=config.src_lang, config=config)
        self.trg_vocab = Vocab(lang=config.trg_lang, config=config)
        self.list_ref_vocab = Vocab(lang=config.trg_lang, config=config)

        self.is_train = (dataset == "train")
        self.max_seq_len = config.max_seq_len
        self.sort_in_batch = sort_in_batch
        self.is_shuffle = shuffle
        self.is_sample = sample
        self.batch_size = batch_size
        
        self.num_batches = 0
        #self.embed_path ='/Users/ftahmasebian/Documents/Projects/VarientNMT/pretrain_embedding/glove.840B.300d.txt'
        #self.embed_path =r'/home/farnaz/Research/Fake review/Torch_VNMT-master/Torch_VNMT-master/Torch_VNMT/pretrain_embedding/glove.840B.300d.txt'
        self.embed_path = r'C:\Users\x1\Documents\Farnaz\Research\Fake review\Torch_VNMT-master\Torch_VNMT-master\Torch_VNMT\pretrain_embedding\glove.840B.300d.txt'


    def set_vocab(self, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    
    def load(self, vocab_cache=None):
        if not vocab_cache and self.is_train:
            self.examples.read()
            self.src_vocab.make_vocab([x[0] for x in self.examples])
            self.trg_vocab.make_vocab([x[1] for x in self.examples])
            #self.list_ref_vocab = self.trg_vocab #.make_vocab([s for x in self.examples for s in x[2]])
            self.examples.numericalize(
                src_w2id=self.src_vocab.stoi, 
                trg_w2id=self.trg_vocab.stoi)
            
            
            
            self.src_vocab.load_pretrained_embedding(
                self.embed_path, self.config.src_embed_dim,cfg.src_vocab_embeded) 
            
            print("Source vocab is loaded")
            #self._add_unknown_words(self.config.src_embed_dim)
            self.trg_vocab.load_pretrained_embedding(
                self.embed_path, self.config.trg_embed_dim, cfg.trg_vocab_embeded) 
            #self._add_unknown_words(self.config.trg_embed_dim)
            #self.trg_vocab.load_pretrained_embedding(
            #    self.embed_path, self.config.trg_embed_dim, cfg.trg_vocab_embeded) 

            print("Target vocab is loaded")

        if not self.is_train:
            print("It is not training")
            self.examples.read()
            assert len(self.src_vocab) > 0
            self.examples.numericalize(
                src_w2id=self.src_vocab.stoi, 
                trg_w2id=self.trg_vocab.stoi)
             
            self.src_vocab.load_pretrained_embedding_not_train_data(
                self.embed_path, self.config.src_embed_dim) 
            #self._add_unknown_words()
            print("Source vocab in not training file is loaded")

            self.trg_vocab.load_pretrained_embedding_not_train_data(
                self.embed_path, self.config.trg_embed_dim) 
            print("Target vocab in not training file is loaded")

        self.num_batches = math.ceil(len(self.examples)/self.batch_size)

    def _pad_list(self,lists, max_L):
        while len(lists) < max_L:
            lists.append([])
        return lists
    
    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):
        #TODO: There is a problem here, In the activity lists, the length of some 
        # sentences are max_L+1 :(
        sentence = sentence[:(max_L-2)]
        if add_bos:
            sentence = [w2id[BOS_WORD]] + sentence
        if add_eos:
            sentence =  sentence + [w2id[EOS_WORD]]
        if len(sentence) < max_L:
            sentence = sentence + [w2id[PAD_WORD]] * (max_L-len(sentence))
        return [x for x in sentence]
  

    def pad_seq_pair(self, samples):
        if self.sort_in_batch:
            samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
        pairs = [pair for pair in samples]
        
        
        src_Ls = [len(pair[0])+2 for pair in pairs]
        trg_Ls = [len(pair[1])+2 for pair in pairs]
        #list_ref_Ls = [len(pair[2]) for pair in pairs]
        
        max_trg_Ls = max(trg_Ls)
        max_src_Ls = max(src_Ls)
        #max_list_ref_Ls = max(list_ref_Ls)
        
        src = [self._pad(
            src, max_src_Ls, self.src_vocab.stoi, add_bos=True, add_eos=True) for src,_ in pairs]
        trg = [self._pad(
            trg, max_trg_Ls, self.trg_vocab.stoi, add_bos=True, add_eos=True) for _,trg in pairs]
#        src = [self._pad(
#            src, max_src_Ls, self.src_vocab.stoi, add_bos=True, add_eos=True) for src,_,_ in pairs]
#        trg = [self._pad(
#            trg, max_trg_Ls, self.trg_vocab.stoi, add_bos=True, add_eos=True) for _,trg,_ in pairs]
#
#        full_list_ref = [self._pad_list(list_ref, max_list_ref_Ls) for _,_,list_ref in pairs]
#        expand_list_ref =[]
#        for list_ref in full_list_ref:    
#            new_list_ref = []
#            for activity in list_ref:
#                new_activity = self._pad(activity, max_trg_Ls, self.trg_vocab.stoi, add_bos=True, add_eos=True)
#                new_list_ref.append(new_activity)
#            expand_list_ref.append(new_list_ref)
#
#        
#        numpy_list_ref = np.array(expand_list_ref,dtype=np.int64)
#        list_ref_tensor = torch.from_numpy(numpy_list_ref)
        
        batch = Batch()
        if 1==0:
            batch.src = torch.LongTensor(src).transpose(0, 1).cuda()
            batch.trg = torch.LongTensor(trg).transpose(0, 1).cuda()            
#            batch.list_ref = torch.LongTensor(list_ref_tensor).transpose(0,2).cuda()
                       
            batch.src_Ls = torch.LongTensor(src_Ls).cuda()
            batch.trg_Ls = torch.LongTensor(trg_Ls).cuda()
            #batch.list_ref = torch.LongTensor(list_ref).cuda()

        else: 
            batch.src = torch.LongTensor(src).transpose(0, 1)
            batch.src =batch.src.to(torch.device("cpu"))
            batch.trg = torch.LongTensor(trg).transpose(0, 1)
            batch.trg =batch.trg.to(torch.device("cpu"))
            #list_ref_tensor.transpose(0,2).shape : 32*275*128 or should it be 275*32*128
#            batch.list_ref = torch.LongTensor(list_ref_tensor).transpose(0,2)
#            batch.list_ref = batch.list_ref.to(torch.device("cpu"))
            
            batch.src_Ls = torch.LongTensor(src_Ls)
            batch.src_Ls = batch.src_Ls.to(torch.device("cpu"))
            batch.trg_Ls = torch.LongTensor(trg_Ls)
            batch.trg_Ls = batch.trg_Ls.to(torch.device("cpu"))
        return batch


    def __iter__(self):
        if self.is_shuffle:
            random.shuffle(self.examples)    
        total_num = len(self.examples)
        for i in range(self.num_batches): 
            if self.is_sample:
                samples = random.sample(self.examples, self.batch_size)
            else:
                samples = self.examples[i * self.batch_size: \
                        min(total_num, self.batch_size*(i+1))]
            yield self.pad_seq_pair(samples)

class Batch(object):
    def __init__(self):
        self.src = None
        self.trg = None
#        self.list_ref = None
        self.src_Ls = None
        self.trg_Ls = None
#        self.list_ref_Ls = None
        
    def __len__(self):
        return self.src_Ls.size(0)
    @property
    def batch_size(self):
        return self.src_Ls.size(0)
    

