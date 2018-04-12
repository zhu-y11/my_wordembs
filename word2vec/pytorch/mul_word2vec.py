# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  10/04/2018 
The code is borrowed from 
https://github.com/Adoni/word2vec_pytorch/
https://github.com/ray1007/pytorch-word2vec/
"""

#************************************************************
# Imported Libraries
#************************************************************
import time
import sys
import numpy as np
from multiprocessing import set_start_method

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn as nn

from args import create_args
from input_data import InputData
from skip_gram import SkipGramModel

np.random.seed(1234)
torch.manual_seed(1234)


class Word2Vec:
  def __init__(self, args):
    # data class
    self.data           = InputData(args.train, args.min_count, args.thread)

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
    self.thread         = args.thread
    self.use_cuda       = args.cuda

    print('Initializing models...')
    self.init_model(args)
    if self.use_cuda:
        self.model.cuda()
    self.model.share_memory()


  def init_model(self, args):
    if args.cbow == 0:
      if self.lr == -1.0:
        self.lr = 0.025
      if self.load_model is not None:
        print('Loading model from: {}...'.format(self.load_model)) 
        self.model = torch.load(self.load_model) 
        self.model.train()
      else: 
        self.model = SkipGramModel(self.data.vocab_size, self.emb_dim)



def train(w2v):
  neg_idxs = list(range(w2v.data.vocab_size))
  # total count
  word_tot_count = mp.Value('L', 0)
  real_word_ct = mp.Value('L', 0)
  processes = []
  for p_id in range(w2v.thread):
    p = mp.Process(target = train_process, args = (p_id, real_word_ct, word_tot_count, neg_idxs, w2v))
    processes.append(p)
    p.start()
  for p in processes:
    p.join()
  print('\nOutput to file: {}\nSave model to: {}'.format(ft.outfile, ft.save_model)) 
  w2v.model.save_embedding(w2v.data, w2v.outfile, w2v.save_model, w2v.use_cuda)


def train_process(p_id, real_word_ct, word_tot_count, neg_idxs, w2v):  
  text = get_thread_text(p_id, w2v)
  with real_word_ct.get_lock():
    real_word_ct.value += sum([len([w for w in line.strip().split() if w in w2v.data.word2idx]) for line in text]) 
  
  t_start = time.monotonic()
  lr = 0
  # pos pairs for batch training
  word_ct = 0
  prev_word_ct = 0
  pairs = []
  # total loss for a checkpoint
  total_loss = torch.Tensor([0])
  # optimizer
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, w2v.model.parameters()), lr = w2v.lr)
  for i in range(w2v.iters):
    for line in text:
      linevec_idx = [w2v.data.word2idx[w] for w in line.strip().split() if w in w2v.data.word2idx]
      word_ct += len(linevec_idx)
      # subsampling
      linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= w2v.sub_samp_probs[w_idx]]
      if len(linevec_idx) > 1:
        pairs += w2v.data.get_batch_pairs(linevec_idx, w2v.win_size)
      if len(pairs) < w2v.bs:
        # not engouh training pairs
        continue

      total_loss += train_batch(w2v, optimizer, pairs[:w2v.bs], w2v.bs, neg_idxs) 
      pairs = pairs[w2v.bs:]
      with word_tot_count.get_lock():
        word_tot_count.value += word_ct
      word_ct = 0

      if word_tot_count.value - prev_word_ct > 200 * w2v.bs:
        lr = w2v.lr * (1 - word_tot_count.value / (w2v.iters * real_word_ct.value))
        if lr < 0.0001 * w2v.lr:
          lr = 0.0001 * w2v.lr
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f" 
                          % (lr, 
                          word_tot_count.value / (w2v.iters * real_word_ct.value) * 100, 
                          total_loss, 
                          word_tot_count.value / (time.monotonic() - t_start)))  
        sys.stdout.flush()
        total_loss = torch.Tensor([0])
        prev_word_ct = word_tot_count.value
               
  if pairs:
    while pairs:
      total_loss += train_batch(w2v, optimizer, pairs[:w2v.bs], len(pairs[:w2v.bs]), neg_idxs)
      pairs = pairs[w2v.bs:]
    with word_tot_count.get_lock():
      word_tot_count.value += word_ct
    sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f" 
                          % (lr, 
                          word_tot_count.value / (w2v.iters * real_word_ct.value) * 100, 
                          total_loss, 
                          word_tot_count.value / (time.monotonic() - t_start)))  
    sys.stdout.flush()


def get_thread_text(p_id, w2v):
  start_pos = w2v.data.start_pos[p_id]
  end_pos = w2v.data.end_pos[p_id]
  with open(w2v.data.infile) as fin:
    if p_id == 0:
      fin.seek(0)
    else:
      fin.seek(start_pos + 1)
    if end_pos is None:
      text = fin.read().strip().split('\n')
    else:
      nbytes = end_pos - start_pos + 1
      text = fin.read(nbytes).strip().split('\n')
  return text


def train_batch(w2v, optimizer, pairs, bs, neg_idxs):
  pos_u, pos_v = map(list, zip(*pairs))
  neg_v = np.zeros((bs, w2v.neg_n), dtype = int)
  neg_v = np.random.choice(neg_idxs, (bs, w2v.neg_n), p = w2v.data.neg_sample_probs)

  pos_u = Variable(torch.LongTensor(pos_u))
  pos_v = Variable(torch.LongTensor(pos_v))
  neg_v = Variable(torch.LongTensor(neg_v))
  if w2v.use_cuda:
    pos_u = pos_u.cuda()
    pos_v = pos_v.cuda()
    neg_v = neg_v.cuda() 

  optimizer.zero_grad()
  loss = w2v.model(pos_u, pos_v, neg_v)
  loss.backward()
  optimizer.step()
  return loss.cpu().data


if __name__ == '__main__':
  set_start_method('spwan')
  args = create_args()
  w2v = Word2Vec(args)
  train(w2v) 
