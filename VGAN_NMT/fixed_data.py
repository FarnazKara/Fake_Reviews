#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:25:43 2019

@author: ftahmasebian
"""

"""
Configuration script. Stores variables and settings used across application
"""
import os

CUR_DIR = os.getcwd()  
BASE_DIR = os.path.join(CUR_DIR, 'data')
#RAW_FILE = os.path.join(BASE_DIR , 'Sample-crt-cmp-wa-job.csv')
#RAW_FILE = os.path.join(BASE_DIR , 'AI-Res-sampledata-big.csv')  
#RAW_FILE = os.path.join(BASE_DIR , 'aires-crt-cmp-activities.csv')
RAW_FILE = os.path.join(BASE_DIR , '100k_reviews.csv')
TRAIN_X_FILE = os.path.join(BASE_DIR , 'train.title.csv')
TRAIN_Y_FILE = os.path.join(BASE_DIR , 'train.activity.csv')
VAL_Y_FILE = os.path.join(BASE_DIR , 'dev.activity.csv')
VAL_X_FILE = os.path.join(BASE_DIR , 'dev.title.csv')
TEST_X_FILE = os.path.join(BASE_DIR , 'test.title.csv')
TEST_Y_FILE = os.path.join(BASE_DIR , 'test.activity.csv')
src_vocab_embeded = os.path.join(BASE_DIR , 'src_embed_vocab.pickle')
trg_vocab_embeded = os.path.join(BASE_DIR , 'trg_embed_vocab.pickle')


GLOVE_EMBEDDING = os.path.join(BASE_DIR , 'glove.6B.50d.txt')
WORD2VEC_EMBEDDING = os.path.join(BASE_DIR ,'google.txt')

TRAINING_SPLIT = 0.05
VALIDATION_SPLIT = 0.05

MAX_SEQUENCE_LENGTH = 250
MAX_NB_WORDS = 12000
WORD_EMBEDDING_DIM = 300

CUT_OFF_WORDS = 5
unk_threshold = 10 # help = 'minimum word frequency to be in dictionary'

Review_COLUMNS = 'review'#'job_activities'
CATEGORY_COLUMN = 'categories'
CITY_COLUMN = 'city'
STATE_COLUMN = 'state'
STAR_COLUMN = 'stars'
COMPNY_COLUMN = 'name'

#'categories', 'city', 'name', 'review', 'stars', 'state'

SELECTED_COLUMNS =[STAR_COLUMN,COMPNY_COLUMN, CITY_COLUMN, STATE_COLUMN, CATEGORY_COLUMN, Review_COLUMNS]

X_COLUMNS = [STAR_COLUMN,COMPNY_COLUMN, CITY_COLUMN, STATE_COLUMN, CATEGORY_COLUMN]
TARGET_COLUMN = [Review_COLUMNS]


"""
# default training settings
data_dir = 'data/negotiate' # data corpus directory
nembed_word = 256 # size of word embeddings
nembed_ctx = 64 # size of context embeddings
nhid_lang = 256 # size of the hidden state for the language model
nhid_ctx = 64 # size of the hidden state for the context model
nhid_strat = 64 # size of the hidden state for the strategy model
nhid_attn = 64 # size of the hidden state for the attention module
nhid_sel = 64 # size of the hidden state for the selection module
lr = 20.0 # initial learning rate
min_lr = 1e-5 # min thresshold for learning rate annealing
decay_rate = 9.0 # decrease learning rate by this factor
decay_every = 1 # decrease learning rate after decay_every epochs
momentum = 0.0 # momentum for SGD
nesterov = False # enable Nesterov momentum
clip = 0.2 # gradient clipping
dropout = 0.5 # dropout rate in embedding layer
init_range = 0.1 #initialization range
max_epoch = 30 # max number of epochs
bsz = 25 # batch size
unk_threshold = 20 # minimum word frequency to be in dictionary
temperature = 0.1 # temperature
sel_weight = 1.0 # selection weight
seed = 1 # random seed
cuda = False # use CUDA
plot_graphs = False # use visdom
rnn_ctx_encoder = False # Whether to use RNN for encoding the context

#fixes
rl_gamma = 0.95
rl_eps = 0.0
rl_momentum = 0.0
rl_lr = 20.0 
rl_clip = 0.2
rl_reinforcement_lr = 20.0
rl_reinforcement_clip = 0.2
rl_bsz = 25
rl_sv_train_freq = 4
rl_nepoch = 4
rl_score_threshold= 6
verbose = True
rl_temperature = 0.1
"""