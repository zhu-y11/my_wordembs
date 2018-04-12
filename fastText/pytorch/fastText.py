# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  12/04/2018 
fastText Pytorch Implementation
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


class FastText:
  def __init__(self, args):
    # data class
    self.data           = InputData(args.train, args.min_count, args.minn, args.maxn)

    self.outfile        = args.output                                       
    self.save_model     = args.save_model
    self.load_model     = args.load_model
    self.emb_dim        = args.size                                         
    self.bs             = args.batch_size
    self.win_size       = args.window
    self.iters          = args.iter
    self.lr             = args.lr
    self.neg_n          = args.negative
    self.sub_samp_th    = args.sample
    #subsampling,  prob reserving the word
    self.sub_samp_probs = np.sqrt(self.sub_samp_th / self.data.idx2freq)      
    self.use_cuda       = args.cuda

    print('Initializing models...')
    self.init_model(args)
    if self.use_cuda:
        self.model.cuda()


  def init_model(self, args):
    if args.cbow == 0:
      if self.lr == -1.0:
        self.lr = 0.05
      if self.load_model is not None:
        print('Loading model from: {}...'.format(self.load_model))
        self.model = torch.load(self.load_model) 
        self.model.train()
      else: 
        self.model = SkipGramModel(self.data.ngm_size, self.data.vocab_size, self.emb_dim)



def train(ft):
  t_start = time.monotonic()
  lr = 0
  neg_idxs = list(range(ft.data.vocab_size))
  word_ct = 0 
  # total count
  word_tot_count = 0
  # pos pairs for batch training
  pairs = []
  # numbers of batches
  bs_n = 0
  # total loss for a checkpoint
  total_loss = torch.Tensor([0]) 
  # optimizer
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, ft.model.parameters()), lr = ft.lr)
  #optimizer = optim.SparseAdam(filter(lambda p: p.requires_grad, ft.model.parameters()), lr = ft.lr)
  for i in range(ft.iters):
    with open(ft.data.infile, 'r') as fin:
      for line in fin:
        linevec_idx = [ft.data.word2idx[w] for w in line.strip().split() if w in ft.data.word2idx]
        word_ct += len(linevec_idx)
        # subsampling
        linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= ft.sub_samp_probs[w_idx]]
        if len(linevec_idx) > 1:
          pairs += ft.data.get_batch_pairs(linevec_idx, ft.win_size)
        if len(pairs) < ft.bs:
          # not engouh training pairs
          continue

        total_loss += train_batch(ft, optimizer, pairs[:ft.bs], ft.bs, neg_idxs) 
        pairs = pairs[ft.bs:]
        bs_n += 1

        if word_ct > 1e4:
          word_tot_count += word_ct
          word_ct = 0
          lr = 0
          lr = ft.lr * (1 - word_tot_count / (ft.iters * ft.data.word_ct))
          if lr < 0.0001 * ft.lr:
            lr = 0.0001 * ft.lr
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr
          sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f" 
                            % (lr, 
                            word_tot_count / (ft.iters * ft.data.word_ct) * 100, 
                            total_loss / bs_n, 
                            word_tot_count / (time.monotonic() - t_start)))  
          sys.stdout.flush()
          total_loss = torch.Tensor([0])
          bs_n = 0
       
  if pairs:
    while pairs:
      total_loss += train_batch(ft, optimizer, pairs[:ft.bs], len(pairs[:ft.bs]), neg_idxs)
      bs_n += 1
      pairs = pairs[ft.bs:]
    word_tot_count += word_ct
    sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f"
                          % (lr, 
                          word_tot_count / (ft.iters * ft.data.word_ct) * 100, 
                          total_loss / bs_n, 
                          word_tot_count / (time.monotonic() - t_start)))
    sys.stdout.flush()

  print('\nOutput to file: {}\nSave model to: {}'.format(ft.outfile, ft.save_model))
  ft.model.save_embedding(ft.data, ft.outfile, ft.save_model, ft.use_cuda)


def train_batch(ft, optimizer, pairs, bs, neg_idxs):
  pos_u, pos_v = map(list, zip(*pairs))
  neg_v = np.zeros((bs, ft.neg_n), dtype = int)
  neg_v = np.random.choice(neg_idxs, (bs, ft.neg_n), p = ft.data.neg_sample_probs)

  pos_u = get_seqs(ft, pos_u)
  pos_v = Variable(torch.LongTensor(pos_v))
  neg_v = Variable(torch.LongTensor(neg_v))

  if ft.use_cuda:
    pos_u = pos_u.cuda()
    pos_v = pos_v.cuda()
    neg_v = neg_v.cuda() 

  optimizer.zero_grad()
  loss = ft.model(pos_u, pos_v, neg_v)
  loss.backward()
  optimizer.step()
  return loss.cpu().data


def get_seqs(ft, seqs):
  new_seqs = []
  for seq in seqs:
    new_seqs.append(ft.data.wdidx2ngidx[seq])
  return Variable(torch.LongTensor(np.array(new_seqs)), requires_grad = False)


if __name__ == '__main__':
  args = create_args()
  ft = FastText(args)
  train(ft) 
