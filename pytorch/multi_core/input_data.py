# -*- coding: UTF-8 -*-
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
from collections import defaultdict
import os
import sys
import numpy as np

import pdb


class InputData(object):
  def __init__(self, infile, min_count, thread_n = 1):
    """
    vocab_file: file containing word freq pairs
    """
    self.infile = infile
    self.vocab_file = self.infile + '.dict'
    self.min_count = min_count

    # split the file into n parts, n = thread_number
    self.start_pos = []
    self.end_pos = []
    self.get_pos(thread_n)

    # generate word -> freq vocab_file 
    if not os.path.exists(self.vocab_file):
      print('Did not found vocabulary file, generating vocabulary file...')
      self.gen_vocab()

    print('Reading vocabulary file...')
    self.word2idx = {}
    self.idx2word = {}
    self.idx2ct = None
    self.idx2freq = None
    self.read_vocab() 
    self.vocab_size = len(self.word2idx)
    self.word_ct = self.idx2ct.sum()
    print('Vocabulary size: {}'.format(self.vocab_size))
    print("Words in train file: {}".format(self.word_ct))

    self.sample_table_size = 1e8
    self.neg_sample_probs = None
    self.init_sample_table()


  def gen_vocab(self):
    word2ct = defaultdict(int)
    line_n = len(open(self.infile, 'r').readlines())
    with open(self.infile, 'r') as fin:
      for i in range(line_n):
        sys.stdout.write('{}/{}\r'.format(i, line_n))
        sys.stdout.flush()
        line = fin.readline()
        linevec = line.strip().split(' ')
        for w in linevec:
          word2ct[w] += 1 
    with open(self.vocab_file, 'w') as fout:
      # sort the pair in descending order
      for w, c in sorted(word2ct.items(), key = lambda x: x[1], reverse = True):
        fout.write('{}\t{}\n'.format(w, c))
  

  def get_pos(self, thread_n):
    file_size = os.path.getsize(self.infile) #size of file (in bytes)
    #break the file into 20 chunks for processing.
    file_step = file_size // thread_n if file_size % thread_n == 0 else file_size // thread_n + 1
    initial_chunks = range(1, file_size, file_step)
    with open(self.infile, 'r') as fin:
      self.start_pos = sorted(set([self.newlinebefore(fin, i) for i in initial_chunks]))
    assert(len(self.start_pos) == thread_n)
    self.end_pos = [i - 1 for i in self.start_pos] [1:] + [None]


  def newlinebefore(self, f, n):
    f.seek(n)
    c = f.read(1)
    while c != '\n' and n > 0:
      n -= 1
      f.seek(n)
      try:
        c = f.read(1)
      except UnicodeDecodeError:
        continue
    return n


  def read_vocab(self):
    """
    #get word-> freq from vocab
    """
    word2freq = defaultdict(int)
    line_n = len(open(self.vocab_file, 'r').readlines())
    with open(self.vocab_file, 'r') as fin:
      for line in fin:
        linevec = line.strip().split()
        assert(len(linevec) == 2)
        word2freq[linevec[0].strip()] = int(linevec[1])

    idx = 0
    self.idx2ct = {}
    for w, c in word2freq.items():
      if c < self.min_count:
        # word2freq is already sorted according to count
        break
      self.word2idx[w] = idx 
      self.idx2word[idx] = w
      self.idx2ct[idx] = c
      idx += 1
    self.idx2ct = np.array(list(self.idx2ct.values()))
    self.idx2freq = self.idx2ct / self.idx2ct.sum()  


  def init_sample_table(self):
    pow_ct = np.array(list(self.idx2ct)) ** 0.75
    words_pow = sum(pow_ct)
    self.neg_sample_probs = pow_ct / words_pow

  
  def get_batch_pairs(self, linevec_idx, win_size):
    pairs = []
    for i, w in enumerate(linevec_idx):
      # dynamic window size [1, win_size]
      actual_win_size = np.random.randint(5) + 1
      # get context according to window size
      context = linevec_idx[max(0, i - actual_win_size): i] + linevec_idx[i + 1: i + 1 + actual_win_size]
      for c in context:
        pairs.append((w, c))
    return pairs
    '''
    pair_value_probs = pair_values / np.sum(pair_values)
    batch_keys = np.random.choice(pair_key_idxs, bs, p = pair_value_probs)
    batch_in = []
    for k in batch_keys:
      batch_in.append(pair_keys[k])
      pair_values[k] = max(0, pair_values[k] - 1)
    return map(list, zip(*batch_in))
    '''

if __name__ == '__main__':
  test = InputData('de.sent.small', 5)
