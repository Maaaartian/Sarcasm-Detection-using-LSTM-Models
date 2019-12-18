#!/usr/bin/env python
# coding: utf-8

import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors, GloVe

def load_dataset(dataset_path):
	# input data field for preprocessing
	TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, include_lengths=True, batch_first=True, fix_length = 80) ### fix_length = 80
	LABEL = data.Field(sequential=False, use_vocab = False, is_target = True) # dtype = torch.float64

	# load dataset from .csv files
	train_data, valid_data, test_data = data.TabularDataset.splits(
                                path=dataset_path, train='train.csv', validation='val.csv', test='test.csv',
                                format='csv', skip_header=True,
                                fields=[('label', LABEL), ('comment', TEXT), ('author', None), ('parent_comment', TEXT)]) 
                                ### ('parent_comment', TEXT)

	# build vocabulary from the training set
	TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
	LABEL.build_vocab(train_data) # is this necessary for training?
	'''
	print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
	print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
	print ("Label Length: " + str(len(LABEL.vocab)))
	'''

	# create iterators for each set -- set batch size = 8
	train_iter, valid_iter, test_iter = data.BucketIterator.splits(
                                	(train_data, valid_data, test_data), batch_size=8, 
                                	sort_key=lambda x: len(x.comment), repeat=False, shuffle=True)
									### (lambda x: len(x.comment + x.parent_comment))

	return TEXT, train_iter, valid_iter, test_iter