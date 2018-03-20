# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  20/03/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from args import create_args
from input_data import InputData
from skip_gram import SkipGramModel

import pdb

np.random.seed(1234)
torch.manual_seed(1234)


class Word2Vec:
  def __init__(self, args):
    # data class
    self.data           = InputData(args.train, args.min_count)

    self.outfile        = args.output                                       
    self.emb_dim        = args.size                                         
    self.bs             = args.batch_size
    self.win_size       = args.window
    self.iters          = args.iter
    self.lr             = args.lr
    self.neg_n          = args.negative
    self.sub_samp_th    = args.sample
    self.sub_samp_probs = np.sqrt(self.sub_samp_th / self.data.idx2freq)  #subsampling,  prob reserving the word
    self.use_cuda       = args.cuda

    print('Initializing models...')
    self.init_model(args)
    if self.use_cuda:
        self.model.cuda()
    self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
    #self.optimizer = optim.SparseAdam(self.model.parameters(), lr = self.lr)


  def init_model(self, args):
    if args.cbow == 0:
      if self.lr == -1.0:
        self.lr = 0.025
      self.model = SkipGramModel(self.data.vocab_size, self.emb_dim)


  def train(self):
    """
    Multiple training.
    """
    t_start = time.monotonic()
    for i in range(self.iters):
      batch_ct = 0
      # store extra pos pairs for next batch
      extra_pairs = []
      with open(self.data.infile, 'r') as fin:
        for line in fin:
          linevec_idx = [self.data.word2idx[w] for w in line.strip().split() if w in self.data.word2idx]
          # subsampling
          linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= self.sub_samp_probs[w_idx]]
          if len(linevec_idx) == 1:
            continue
          pairs = self.data.get_batch_pairs(linevec_idx, self.win_size)
          pdb.set_trace()

      """
      pair_keys = list(self.data.train_pairs.keys())
      pair_key_idxs = list(range(len(pair_keys)))
      pair_values = np.array(list(self.data.train_pairs.values()))
      for j in range(batch_n):
        pos_u, pos_v = self.data.get_batch_pairs(self.bs, pair_keys, pair_key_idxs, pair_values)
        sub_samp_idxs = np.nonzero((np.random.random_sample((self.bs, )) - self.sub_samp[pos_u]) > 0)[0]
        pos_u = np.array(pos_u)[sub_samp_idxs]
        pos_v = np.array(pos_v)[sub_samp_idxs]
        neg_v = np.random.choice(neg_idxs, (pos_u.shape[0], self.neg_n), p = self.data.sample_probs) 
        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))

        if self.use_cuda:
          pos_u = pos_u.cuda()
          pos_v = pos_v.cuda()
          neg_v = neg_v.cuda() 

        self.optimizer.zero_grad()
        loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
        loss.backward()
        self.optimizer.step()

        print("epoch {}, batch {}/{}\nloss = {:.5f}, lr: {:.6}".format(i + 1, j + 1, batch_n, loss.data[0] / pos_u.size()[0], self.optimizer.param_groups[0]['lr']))
        if j > 0 and j % 100 == 0:
          lr = lr * (1.0 - 0.01)
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
      """


if __name__ == '__main__':
  args = create_args()
  w2v = Word2Vec(args)
  w2v.train() 
