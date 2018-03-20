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
from collections import defaultdict
from tqdm import tqdm
import os
import sys
import numpy as np

import pdb


class InputData(object):
  def __init__(self, infile, min_count):
    """
    vocab_file: file containing word freq pairs
    """
    self.infile = infile
    self.vocab_file = self.infile + '.dict'
    self.min_count = min_count

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
      for line in tqdm(fin, total = line_n):
        linevec = line.strip().split(' ')
        for w in linevec:
          word2ct[w] += 1
    self.vocab_file = self.infile + '.dict'
    with open(self.vocab_file, 'w') as fout:
      # sort the pair in descending order
      for w, c in sorted(word2ct.items(), key = lambda x: x[1], reverse = True):
        fout.write('{}\t{}\n'.format(w, c))


  def read_vocab(self):
    """
    get word-> freq from vocab
    """
    word2freq = defaultdict(int)
    line_n = len(open(self.vocab_file, 'r').readlines())
    with open(self.vocab_file, 'r') as fin:
      for line in tqdm(fin, total = line_n):
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
    for i, w in enumerate(linevec_idx):
      pdb.set_trace()
      #actual_win_size = 
      # get context according to window size
      context = linevec_idx[max(0, i - win_size): i] + linevec_idx[i + 1: i + 1 + win_size]
      for c in context:
        self.train_pairs[(w, c)] += 1
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
  test = InputData('/mnt/hdd/yz568/data/wiki_txt/de.sent', 5)
