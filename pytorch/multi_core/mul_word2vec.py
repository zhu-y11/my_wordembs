## -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  27/03/2018 
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
from multiprocessing import set_start_method

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.nn as nn

from args import create_args
from input_data import InputData
from skip_gram import SkipGramModel

import pdb

np.random.seed(1234)
torch.manual_seed(1234)

class Word2Vec:
  def __init__(self, args):
    # data class
    self.data           = InputData(args.train, args.min_count, args.thread)

    self.outfile        = args.output                                       
    self.emb_dim        = args.size                                         
    self.bs             = args.batch_size
    self.win_size       = args.window
    self.iters          = args.iter
    self.lr             = args.lr
    self.neg_n          = args.negative
    self.sub_samp_th    = args.sample
    # subsampling, prob reserving the word
    self.sub_samp_probs = np.sqrt(self.sub_samp_th / self.data.idx2freq)      
    self.thread         = args.thread
    self.use_cuda       = args.cuda

    print('Initializing models...')
    self.init_model(args)
    self.model.share_memory()
    #self.model = nn.DataParallel(self.model)
    if self.use_cuda:
        self.model = self.model.cuda()


  def init_model(self, args):
    if args.cbow == 0:
      if self.lr == -1.0:
        self.lr = 0.025
      self.model = SkipGramModel(self.data.vocab_size, self.emb_dim)



def train(model, args):
  neg_idxs = list(range(model.data.vocab_size))
  # total count
  word_tot_count = mp.Value('L', 0)
  real_word_ct = mp.Value('L', 0)
  processes = []
  for p_id in range(args.thread):
    p = mp.Process(target = train_process, args = (p_id, real_word_ct, word_tot_count, neg_idxs, model, args))
    processes.append(p)
    p.start()
  for p in processes:
    p.join()
  print('\nOutput to file...')
  #model.model.save_embedding(model.data.idx2word, model.outfile, model.use_cuda)


def train_process(p_id, real_word_ct, word_tot_count, neg_idxs, model, args):
  t_start = time.monotonic()
  prev_word_ct = 0
  optimizer = optim.SGD(model.model.parameters(), lr = model.lr)

  start_pos = model.data.start_pos[p_id]
  end_pos = model.data.end_pos[p_id]
  with open(model.data.infile) as fin:
    if p_id == 0:
      fin.seek(0)
    else:
      fin.seek(start_pos + 1)
    if end_pos is None:
      text = fin.read().strip().split('\n')
    else:
      nbytes = end_pos - start_pos + 1
      text = fin.read(nbytes).strip().split('\n')

  with real_word_ct.get_lock():
    real_word_ct.value += sum([len([w for w in line.strip().split() if w in model.data.word2idx]) for line in text])

  for i in range(args.iter):
    #print(i)
    # pos pairs for batch training
    pairs = []
    # store extra pos pairs for next batch
    extra_pairs = []
    word_ct = 0
    last_word_ct = 0
    total_loss = torch.Tensor([0])
    for line in text:
      linevec_idx = [model.data.word2idx[w] for w in line.strip().split() if w in model.data.word2idx]
      word_ct += len(linevec_idx)
      # subsampling
      linevec_idx = [w_idx for w_idx in linevec_idx if np.random.random_sample() <= model.sub_samp_probs[w_idx]]
      if len(linevec_idx) == 1:
        continue
      pairs += model.data.get_batch_pairs(linevec_idx, args.window)
      if len(pairs) < args.batch_size:
        # not engouh training pairs
        continue 
      extra_pairs = pairs[args.batch_size:]
      pairs = pairs[:args.batch_size]

      pos_u, pos_v, neg_v = gen_batch(model.data.neg_sample_probs, pairs, neg_idxs, args.batch_size, args.negative, args.cuda)
      with word_tot_count.get_lock():
        word_tot_count.value += word_ct - last_word_ct
      last_word_ct = word_ct

      pairs = extra_pairs[:]
      extra_pairs = []

      optimizer.zero_grad()
      #loss = nn.parallel.data_parallel(model.model, (pos_u, pos_v, neg_v))
      loss = model.model(pos_u, pos_v, neg_v)
      #loss = model.model(d[0], d[1], d[2])
      #print('hahaha')
      loss.backward()
      optimizer.step()

      total_loss += loss.cpu().data
      if word_tot_count.value - prev_word_ct > 10000:
        lr = model.lr * (1 - word_tot_count.value / (model.iters * model.data.word_ct))
        if lr < 0.0001 * model.lr:
          lr = 0.0001 * model.lr
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
        sys.stdout.write("\rAlpha: %0.8f, Progess: %0.2f, Loss: %0.5f, Words/sec: %0.1f" % (lr, word_tot_count.value / (model.iters * real_word_ct.value) * 100, total_loss, word_tot_count.value / (time.monotonic() - t_start)))
        sys.stdout.flush()
        total_loss = torch.Tensor([0])
        prev_word_ct = word_tot_count.value

    if pairs:
      #print('here again')
      pos_u, pos_v, neg_v = gen_batch(model.data.neg_sample_probs, pairs, neg_idxs, len(pairs), args.negative, args.cuda)
      #print('here again again again')
      with word_tot_count.get_lock():
        word_tot_count.value += word_ct - last_word_ct

      optimizer.zero_grad()
      #loss = nn.parallel.data_parallel(model.model, (pos_u, pos_v, neg_v))
      loss = model.model(pos_u, pos_v, neg_v)
      #loss = model.model(d[0], d[1], d[2])
      #print('hahaha')
      loss.backward()
      optimizer.step()


def gen_batch(neg_sample_probs, pairs, neg_idxs, bs, neg_n, use_cuda):
  pos_u, pos_v = map(list, zip(*pairs))
  neg_v = np.zeros((bs, neg_n), dtype = int)
  # make sure pos pairs not in any neg pairs
  neg_v_big = np.random.choice(neg_idxs, (bs, 2 * neg_n), p = neg_sample_probs)
  for k in range(neg_v_big.shape[0]):
    neg_v[k] = neg_v_big[k][np.nonzero(neg_v_big[k] != pos_v[k])[0]][:neg_n]
  pos_u = Variable(torch.LongTensor(pos_u))
  pos_v = Variable(torch.LongTensor(pos_v))
  neg_v = Variable(torch.LongTensor(neg_v))
  if use_cuda:
    pos_u = pos_u.cuda()
    pos_v = pos_v.cuda()
    neg_v = neg_v.cuda() 
  return (pos_u, pos_v, neg_v)


if __name__ == '__main__':
  set_start_method('spawn')
  args = create_args()
  w2v = Word2Vec(args)
  train(w2v, args) 
