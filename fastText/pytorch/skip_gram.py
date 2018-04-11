# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  10/04/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
"""

#************************************************************
# Imported Libraries
#************************************************************
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import pdb


class SkipGramModel(nn.Module):
  def __init__(self, ngm_size, vocab_size, emb_dim):
    super(SkipGramModel, self).__init__()
    self.emb_dim = emb_dim
    self.u_embeddings = nn.Embedding(ngm_size + 1, emb_dim, sparse = True, padding_idx = ngm_size)
    self.v_embeddings = nn.Embedding(vocab_size, emb_dim, sparse = True)
    self.init_emb()


  def init_emb(self):
    """
    Initialize embedding weight like word2vec.
    The u_embedding is a uniform distribution in [-0.5/emb_dim, 0.5/emb_dim], 
    and the elements of v_embedding are zeroes.
    """
    initrange = 0.5 / self.emb_dim
    self.u_embeddings.weight.data.uniform_(-initrange, initrange)
    self.u_embeddings.weight.data[-1] = 0
    self.v_embeddings.weight.data.zero_()


  def forward(self, pos_u, pos_v, neg_v):
    emb_pos_u = self.u_embeddings(pos_u)
    emb_pos_v = self.v_embeddings(pos_v)
    emb_neg_v = self.v_embeddings(neg_v)
    
    # sum all the char n-grams up
    emb_pos_u = emb_pos_u.sum(1)

    score = torch.bmm(emb_pos_u.unsqueeze(1), emb_pos_v.unsqueeze(2))
    score = F.logsigmoid(score)

    neg_score = torch.bmm(emb_neg_v, emb_pos_u.unsqueeze(2))
    neg_score = F.logsigmoid(-1 * neg_score)

    return -1 * (torch.sum(score) + torch.sum(neg_score)) / pos_u.size()[0]


  def save_embedding(self, input_data, outfile, model_file, use_cuda):
    """
    Save all embeddings to file.
    As this class only record word id, so the map from id to word has to be transfered from outside.
    """
    torch.save(self, model_file)
    self.eval()
    with open(outfile, 'w') as fout:
      fout.write('{} {}\n'.format(len(input_data.idx2word), self.emb_dim))
      for word_idx, word in input_data.idx2word.items():
        word_idxs = Variable(torch.LongTensor(input_data.wdidx2ngidx[word_idx]))
        if use_cuda:
          word_idxs = word_idxs.cuda()
        word_embs = self.u_embeddings(word_idxs).sum(0).cpu().data.numpy()
        word_embs = ' '.join(map(lambda x: str(x), word_embs))
        fout.write('{} {}\n'.format(word, word_embs))
