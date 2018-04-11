# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  11/04/2018 
The code is borrowed from 
FastText Pytorch Implementation
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import torch


def create_args():
  parser = argparse.ArgumentParser(description = 'word2vec')
  # The following arguments are mandatory
  parser.add_argument("--train", type = str, required = True, 
      help = "training file path")
  parser.add_argument("--output", type = str, default = "vectors.txt",
      help = "output file path")

  # The following arguments for the dictionary are optional
  parser.add_argument("--min_count", type = int, default = 5,
      help = "minimal number of word occurences")
  parser.add_argument("--minn", type = int, default = 3,
      help = "min length of char ngram")
  parser.add_argument("--maxn", type = int, default = 6,
      help = "max length of char ngram")

  #The following arguments for training are optional:
  parser.add_argument("--save_model", type = str, default = "model.pth",
      help = "model file to save")
  parser.add_argument("--load_model", type = str, default = None,
      help = "model file to load")
  parser.add_argument("--size", type = int, default = 300,
      help = "size of word vectors")
  parser.add_argument("--cbow", type = int, default = 0,
      help = "1 for cbow, 0 for skipgram")
  parser.add_argument("--window", type = int, default = 8,
      help = "context window size")
  parser.add_argument("--sample", type = float, default = 1e-5,
      help="subsample threshold")
  parser.add_argument("--negative", type = int, default = 10,
      help = "number of negative samples")
  parser.add_argument("--iter", type = int, default = 5, 
      help = "number of iterations")
  parser.add_argument("--lr", type = float, default = -1.0,
      help = "initial learning rate")
  parser.add_argument("--batch_size", type = int, default = 100, 
      help = "(max) batch size")
  parser.add_argument("--thread", type = int, default = 8, 
      help = "number of threads")
  parser.add_argument("--cuda", action = 'store_true', default = False, 
      help = "enable cuda")

  args = parser.parse_args()
  args.cuda = args.cuda if torch.cuda.is_available() else False
  return args
