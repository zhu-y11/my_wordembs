# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated 
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse

def create_parser():
  parser = argparse.ArgumentParser(description = "word2vec python implementation")
  parser.add_argument('--train', metavar='PATH', required = True, 
      help="--train <file>. Use text data from <file> to train the model")

  parser.add_argument('--output', '-o', metavar='PATH', required = True, 
      help="--output <file>. Use <file> to save the resulting word vectors / word clusters")

  parser.add_argument('--size', type = int, default = 100,
      help="--size <int>. Set max skip length between words (default: %(default)s))")

  parser.add_argument('--window', type = int, default = 5,
      help="--window <int>. Set size of word vectors (default: %(default)s))")

  parser.add_argument('--sample', type = float, default = 1e-3,
      help="--sample <float>. Set threshold for occurrence of words. 
      Those that appear with higher frequency in the training data will be randomly down-sampled (default: %(default)s)), useful range is (0, 1e-5)")

  parser.add_argument('--hs' action="store_true", help = 'Use Hierarchical Softmax')

  parser.add_argument('--negative', type = int, default = 5,
      help="--negative <int>. Number of negative examples (default: %(default)s)), common values are 3 - 10 (0 = not used)")

  parser.add_argument('--threads', type = int, default = 12,
      help="Use <int> threads (default: %(default)s))")

  parser.add_argument('--iter', type = int, default = 5,
      help="Use <int> threads (default: %(default)s))")

  ¦ printf("\t-min-count <int>\n");
  ¦ printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
  ¦ printf("\t-alpha <float>\n");
  ¦ printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
  ¦ printf("\t-classes <int>\n");
  ¦ printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
  ¦ printf("\t-debug <int>\n");
  ¦ printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
  ¦ printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
      ¦ printf("\t-save-vocab <file>\n");
  ¦ printf("\t\tThe vocabulary will be saved to <file>\n");
  ¦ printf("\t-read-vocab <file>\n");
  ¦ printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
  ¦ printf("\t-cbow <int>\n");
  ¦ printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
  ¦ printf("\nExamples:\n");
  ¦ printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -it
er 3\n\n");
  ¦ return 0;
  }
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

