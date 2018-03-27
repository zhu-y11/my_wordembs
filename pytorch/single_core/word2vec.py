# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  21/03/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import numpy as np
import time
import sys

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
    neg_idxs = list(range(self.data.vocab_size))
    # total count
    word_tot_count = 0
    for i in range(self.iters):
      # pos pairs for batch training
      pairs = []
      # store extra pos pairs for next batch
      extra_pairs = []
      word_ct = 0
      last_word_ct = 0
      total_loss = torch.Tensor([0])
      with open(self.data.infile, 'r') as fin:
        for line in fin:
          #t1 = time.time()
          linevec_idx = [self.data.word2idx[w] for w in line.strip().split() if w in self.data.word2idx]
          word_ct += len(linevec_idx)
          # subsampling
          linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= self.sub_samp_probs[w_idx]]
          if len(linevec_idx) == 1:
            continue
          pairs += self.data.get_batch_pairs(linevec_idx, self.win_size)
          if len(pairs) < self.bs:
            # not engouh training pairs
            continue
          extra_pairs = pairs[self.bs:]
          pairs = pairs[:self.bs]

          total_loss += self.train_batch(pairs, self.bs, neg_idxs)

          pairs = extra_pairs[:]
          extra_pairs = []
 
          if word_ct - last_word_ct > 10000:
            word_tot_count += word_ct - last_word_ct
            last_word_ct = word_ct
            lr = self.lr * (1 - word_tot_count / (self.iters * self.data.word_ct))
            if lr < 0.0001 * self.lr:
              lr = 0.0001 * self.lr
            for param_group in self.optimizer.param_groups:
              param_group['lr'] = lr
            sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.2f" % (lr, word_tot_count / (self.iters * self.data.word_ct) * 100, total_loss, word_tot_count / (time.monotonic() - t_start)))
            sys.stdout.flush()
            total_loss = torch.Tensor([0])
          #print('loop:{}'.format(time.time() - t1))
         
        word_tot_count += word_ct - last_word_ct
        if pairs:
          total_loss += self.train_batch(pairs, len(pairs), neg_idxs)
          sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %f" % (lr, word_tot_count / (self.iters * self.data.word_ct) * 100, total_loss, word_tot_count / (time.monotonic() - t_start)))
          sys.stdout.flush()
        self.model.save_embedding(self.data.idx2word, self.outfile, self.use_cuda)


  def train_batch(self, pairs, bs, neg_idxs):
    #t2 = time.time() 
    pos_u, pos_v = map(list, zip(*pairs))
    neg_v = np.zeros((bs, self.neg_n), dtype = int)
    # make sure pos pairs not in any neg pairs
    neg_v_big = np.random.choice(neg_idxs, (bs, 2 * self.neg_n), p = self.data.neg_sample_probs)
    for k in range(neg_v_big.shape[0]):
      neg_v[k] = neg_v_big[k][np.nonzero(neg_v_big[k] != pos_v[k])[0]][:self.neg_n]

    #print('io: {}'.format(time.time() - t2))
    #t3 = time.time()

    pos_u = Variable(torch.LongTensor(pos_u))
    pos_v = Variable(torch.LongTensor(pos_v))
    neg_v = Variable(torch.LongTensor(neg_v))
    if self.use_cuda:
      pos_u = pos_u.cuda()
      pos_v = pos_v.cuda()
      neg_v = neg_v.cuda() 


    self.optimizer.zero_grad()
    loss = self.model.forward(pos_u, pos_v, neg_v)
    loss.backward()
    self.optimizer.step()
    #print('network: {}'.format(time.time() - t3))
    #print('function: {}'.format(time.time() - t2))
    return loss.cpu().data



if __name__ == '__main__':
  args = create_args()
  w2v = Word2Vec(args)
  w2v.train() 
