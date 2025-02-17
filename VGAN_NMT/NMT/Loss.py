import torch
import torch.nn as nn
from torch.autograd import Variable

import NMT
import Utils
from NMT.Trainer import Trainer
from NMT.Statistics import Statistics

import torch.nn.functional as F
from Utils.utils import trace

class NMTLoss(nn.Module):
    """
    Standard NMT Loss Computation.
    """
    def __init__(self, config, padding_idx):
        super(NMTLoss, self).__init__()
        self.padding_idx = padding_idx
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, size_average=False)
        self.kld_weight = config.kld_weight
        self.kld_increase = 0.01
    def kld_weight_step(self, epoch, start_increase_kld_at=8):
        pass
        # if self.kld_weight < 0.1 and epoch >= start_increase_kld_at:
        #     self.kld_weight += self.kld_increase
        #     trace("Increase KLD weight to %.2f" % self.kld_weight)

    def compute_batch_loss(self, probs, golds, normalization, kld_loss):
        """Compute the forward loss and backpropagate.

        Args:
          probs (FloatTensor) : distribution of output model `[(trg_len x batch) x V]`
          golds (LongTensor) : target examples
          output (FloatTensor) : output of decoder model `[trg_len x batch x hidden]`

        Returns:
            :`NMT.Statistics`: validation loss statistics

        """
        #TODO: It was golds.view(-1) and go through the loop
        #print("shape of probs is:")
        #print(probs.shape)
        #ce_loss = float('inf')
        chosen = golds #None
        """
        for i,gold in enumerate(golds):
            if self.criterion(probs, gold.contiguous().view(-1)) < ce_loss: 
                chosen = gold
                ce_loss = self.criterion(probs, gold.contiguous().view(-1))
        """
        #print(ce_loss)
        #TODO: Check the average and check by yourself the value of loss
        # Should it be just a value or just a list?
        
        
        #probs: (Length_sentence * BS) * vocab_size
        ce_loss = self.criterion(probs, golds.contiguous().view(-1))

        #TODO: I just add it instead of loss = loss.div(normalization)
        #ce_loss = ce_loss.div(normalization)

        loss = ce_loss + self.kld_weight * kld_loss
        #print("kld_weight is {} and MMD LOSS is {}".format(self.kld_weight , kld_loss))
        loss = loss.div(normalization)
        loss_dict = {
            "CELoss": float(ce_loss)/normalization, 
            "KLDLoss":float(kld_loss)/normalization
            #"CELoss": float(ce_loss), 
            #"KLDLoss":float(kld_loss)
            }
        del ce_loss, kld_loss
        batch_stats = self.create_stats(float(loss), probs,chosen.contiguous().view(-1), loss_dict)
        return loss, batch_stats

    def create_stats(self, loss, probs, golds, loss_dict):
        """
        Args:
            loss (`FloatTensor`): the loss computed by the loss criterion.
            scores (`FloatTensor`): a score for each possible output
            target (`FloatTensor`): true targets

        Returns:
            `Statistics` : statistics for this batch.
        """
        preds = probs.data.topk(1, dim=-1)[1]
        non_padding = golds.ne(self.padding_idx) 
        correct = preds.squeeze().eq(golds).masked_select(non_padding)
        num_words = non_padding.long().sum()
        num_correct = correct.long().sum()
        return Statistics(
            float(loss), int(num_words), 
            int(num_correct), loss_dict)