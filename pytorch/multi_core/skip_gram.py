# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  12/03/2018 
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
  def __init__(self, vocab_size, emb_dim):
    super(SkipGramModel, self).__init__()
    self.emb_dim = emb_dim
    self.u_embeddings = nn.Embedding(vocab_size, emb_dim, sparse = True)
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
    self.v_embeddings.weight.data.zero_()


  def forward(self, pos_u, pos_v, neg_v):
    emb_u = self.u_embeddings(pos_u)
    emb_v = self.v_embeddings(pos_v)
    score = torch.bmm(emb_u.unsqueeze(1), emb_v.unsqueeze(2)).squeeze()
    score = F.logsigmoid(score)

    neg_emb_v = self.v_embeddings(neg_v)
    neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
    neg_score = F.logsigmoid(-1 * neg_score)

    return -1 * (torch.sum(score) + torch.sum(neg_score))


  def save_embedding(self, idx2word, outfile, use_cuda):
    """
    Save all embeddings to file.
    As this class only record word id, so the map from id to word has to be transfered from outside.
    """
    if use_cuda:
        embedding = self.u_embeddings.weight.cpu().data.numpy()
    else:
        embedding = self.u_embeddings.weight.data.numpy()
    with open(outfile, 'w') as fout:
      fout.write('{} {}\n'.format(len(idx2word), self.emb_dim))
      for idx, w in idx2word.items():
        e = embedding[idx]
        e = ' '.join(map(lambda x: str(x), e))
        fout.write('{} {}\n'.format(w, e))



