import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from NMT.Models import NMTModel
from NMT.Models.Decoders import RNNDecoderState

def compute_kld(mu, logvar):
    kld = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
    return kld

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

class VNMTModel(NMTModel):
    """Recent work has found that VAE + LSTM decoders underperforms vanilla LSTM decoder.
    You should use VAE + GRU.
    see Yang et. al, ICML 2017, `Improved VAE for Text Modeling using Dilated Convolutions`.
    """
    def __init__(self, 
                encoder, decoder, 
                src_embedding, trg_embedding, 
                trg_vocab_size, 
                config):
        super(VNMTModel, self).__init__(
                        encoder, decoder, 
                        src_embedding, trg_embedding, 
                        trg_vocab_size, config)
        self.context_to_mu = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        self.context_to_logvar = nn.Linear(
                        config.hidden_size, 
                        config.latent_size)
        self.lstm_state2context = nn.Linear(
                        2*config.hidden_size, 
                        config.latent_size)
    def get_hidden(self, state):
        hidden = None
        if self.encoder.rnn_type == "GRU":
            hidden = state[-1]
        elif self.encoder.rnn_type == "LSTM":
            hidden, context = state[0][-1], state[1][-1]
            hidden = self.lstm_state2context(torch.cat([hidden, context], -1))
        return hidden

    def reparameterize(self, encoder_state):
        """
        context [B x 2H]
        """
        hidden = self.get_hidden(encoder_state)
        mu = self.context_to_mu(hidden)
        logvar = self.context_to_logvar(hidden)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
        else:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            #z = mu
        return z, mu, logvar


    def forward(self, src, src_lengths, trg, trg_lengths=None, decoder_state=None):
        """
        Forward propagate a `src` and `trg` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): source sequence. [L x B x N]`.
            trg (LongTensor): source sequence. [L x B]`.
            src_lengths (LongTensor): the src lengths, pre-padding `[batch]`.
            trg_lengths (LongTensor): the trg lengths, pre-padding `[batch]`.
            dec_state (`DecoderState`, optional): initial decoder state
            z (`FloatTensor`): latent variables
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`nmt.Models.DecoderState`):

                 * decoder output `[trg_len x batch x hidden]`
                 * dictionary attention dists of `[trg_len x batch x src_len]`
                 * final decoder state
        """
        
        # encoding side
        #print("VNMTModel Script")
        #print(type(src))
        #print(src.shape)
        embd = self.src_embedding(src)
        #print(type(embd))
        #print(embd.shape)
        encoder_outputs, encoder_state = self.encoder(
            embd, src_lengths)

        # re-parameterize
        z, mu, logvar = self.reparameterize(encoder_state)
        # encoder to decoder
        decoder_state = self.encoder2decoder(encoder_state)
        
        trg_feed = trg[:-1]
        
        #print(z)
        decoder_input = torch.cat([
                self.trg_embedding(trg_feed), 
                z.unsqueeze(0).repeat(trg_feed.size(0) ,1, 1)],
                -1)
        
        # decoding side
        decoder_outputs, decoder_state, attns = self.decoder(
            decoder_input, encoder_outputs, src_lengths, decoder_state)

        true_samples = Variable(
                torch.randn(z.size(0), z.size(1)),
                requires_grad=False
            )
        
        if torch.cuda.is_available():
            true_samples = true_samples.cuda()
        return decoder_outputs, decoder_state, attns, compute_kld(mu, logvar)
        #return decoder_outputs, decoder_state, attns, compute_mmd(true_samples, z)

  
    
