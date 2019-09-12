"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn
import os

from NMT.Models import NMTModel
from NMT.Models import VNMTModel
from NMT.Models import VRNMTModel
from NMT.Models import PackedRNNEncoder
from NMT.Models import RNNEncoder
from NMT.Models import InputFeedRNNDecoder
from NMT.Models import VarInputFeedRNNDecoder

from torch.nn.init import xavier_uniform
from Utils.DataLoader import PAD_WORD
from Utils.utils import trace, aeq

def make_embeddings(vocab_size, embed_dim, dropout, padding_idx, pretrained_weights):
    """
    Make an Embeddings instance.
    Args:
        config: global configuration settings.
        vocab (Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    #TODO:Correct weights
    #FloatTensor containing pretrained weights
    #print(type(pretrain_weights), pretrain_weights is None)
    #print(pretrain_weights.shape)
    #print(pretrained_weights.type())
    #print(pretrained_weights.shape)
    if torch.cuda.is_available():
        pretrained_weights2 = pretrained_weights.type(torch.cuda.FloatTensor)
    else:
        pretrained_weights2 = pretrained_weights.type(torch.FloatTensor)
    #os.exit()
    embed = nn.Embedding(vocab_size, 
            embed_dim,
            padding_idx=padding_idx, 
            max_norm=None, 
            norm_type=2, 
            scale_grad_by_freq=False, 
            sparse=False)
    embed.weight = nn.Parameter(pretrained_weights2)
    #embedding = nn.Embedding.from_pretrained(pretrain_weights, freeze=True, padding_idx=padding_idx,
    #                max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
    #print(embedding.dim())
    #exit()
    return embed
    
#    return nn.Embedding(vocab_size, 
#            embed_dim,
#            padding_idx=padding_idx, 
#            max_norm=None, 
#            norm_type=2, 
#            scale_grad_by_freq=False, 
#            sparse=False)#, _weight=pretrained_weights)

    # return nn.Embeddings(embed_dim,
    #                   position_encoding=False,
    #                   dropout=dropout,
    #                   word_padding_idx=padding_idx,
    #                   word_vocab_size=vocab_size,
    #                   sparse=True)

def model_factory(config, src_vocab, trg_vocab, train_mode=True, checkpoint=None):
    
    # Make embedding.
    padding_idx = src_vocab.stoi[PAD_WORD]
    src_embeddings = make_embeddings(src_vocab.vocab_size, config.trg_embed_dim, config.dropout, padding_idx,src_vocab.weights)
    #print(src_vocab.itos[4])
    #input = torch.LongTensor([4])
    #print(src_embeddings(input))
    #print(src_vocab.weights.size())
    #src_embeddings.load_state_dict({'weight': src_vocab.weights})

    padding_idx = trg_vocab.stoi[PAD_WORD]
    trg_embeddings = make_embeddings(trg_vocab.vocab_size, config.src_embed_dim, config.dropout, padding_idx, trg_vocab.weights)
    #trg_embeddings.load_state_dict({'weight':trg_vocab.weights})

    # Make NMT Model (= encoder + decoder).
    print("Make Model......")
    if config.system == "NMT":

        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, config.enc_num_layers,
            config.dropout, config.bidirectional)
        decoder = InputFeedRNNDecoder(
            config.rnn_type, config.trg_embed_dim, config.hidden_size,
            config.dec_num_layers, config.attn_type,
            config.bidirectional, config.dropout)
        model = NMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size, config)

    elif config.system == "VNMT":
        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, 
            config.enc_num_layers,
            config.dropout, 
            config.bidirectional)
        
        decoder = InputFeedRNNDecoder(
            config.rnn_type, 
            config.trg_embed_dim+config.latent_size, 
            config.hidden_size, 
            config.dec_num_layers, 
            config.attn_type,
            config.bidirectional, 
            config.dropout)
        
        model = VNMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size,  
            config)

    elif config.system == "VRNMT":
        encoder = PackedRNNEncoder(
            config.rnn_type, 
            config.src_embed_dim, 
            config.hidden_size, 
            config.enc_num_layers,
            config.dropout, 
            config.bidirectional)
        
        decoder = VarInputFeedRNNDecoder(
            config.rnn_type, 
            config.trg_embed_dim,
            config.latent_size, 
            config.hidden_size, 
            config.dec_num_layers, 
            config.attn_type,
            config.bidirectional, 
            config.dropout)
        
        model = VRNMTModel(
            encoder, decoder, 
            src_embeddings, trg_embeddings, 
            trg_vocab.vocab_size,  
            config)
   

    if checkpoint is not None:
        trace('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        
    # Load the model states from checkpoint or initialize them.
    #.requires_grad (boolean indicating whether to calculate gradient for the Variable during backpropagation)
    if train_mode and config.param_init != 0.0:
        trace("Initializing model parameters.")
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
       
    # if hasattr(model.encoder, 'embeddings'):
    #     model.encoder.embeddings.load_pretrained_vectors(
    #                 config.pre_word_vecs_enc, config.fix_word_vecs_enc)
    # if hasattr(model.decoder, 'embeddings'):
    #     model.decoder.embeddings.load_pretrained_vectors(
    #                 config.pre_word_vecs_dec, config.fix_word_vecs_dec)

    if train_mode:
        model.train()
    else:
        model.eval()

    if torch.cuda.is_available():
    #if config.gpu_ids is not None:
        model.cuda()
    else:
        model.cpu()

    return model
