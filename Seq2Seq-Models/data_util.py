import random
from torchtext import data
import csv
import numpy as np
import torch
import spacy

import string

"""
Referenced: https://torchtext.readthedocs.io/en/latest/examples.html
"""

def get_data():
	path = "data_file/sumdata/train/"


	#### GET TRAINSET #####
	with open(path + "train.article.txt") as f:
		train_art = f.read().splitlines()

		f.close()

	with open(path + "train.title.txt") as f:
		train_title = f.read().splitlines()

		f.close()


	t_file = []
	for a, t in zip(train_art, train_title):
		t_file.append([a, t])

	t = random.sample(t_file, 30000)

	tcsv = csv.writer(open("training.csv", "w"))
	for inst in t:
		tcsv.writerow([inst[0]] + [inst[1]])


	#get validation set

	with open(path + "valid.article.filter.txt") as f:
		val_art = f.read().splitlines()

		f.close()

	with open(path + "valid.title.filter.txt") as f:
		val_title = f.read().splitlines()

		f.close()

	#validation = [val_art, val_title]

	v_file = []
	for a, t in zip(val_art, val_title):
		v_file.append([a, t])

	v = random.sample(v_file, 10000)

	vcsv = csv.writer(open("validation.csv", "w"))
	for inst in v:
		vcsv.writerow([inst[0]] + [inst[1]])
		#vcsv.writerow([inst[1]])


def preprocess(load=False):
	#tokenize = data.get_tokenizer("basic_english")
	tokenize = spacy.load('en_core_web_sm')


	def reverse_tokenize(txt):
		return [tok.text for tok in tokenize.tokenizer(txt)][::-1]

	txt = data.Field(sequential=True, tokenize=reverse_tokenize,
	                 init_token = '<sos>', eos_token = '<eos>',  lower=True)
	#sum = data.Field(sequential=True, tokenize=tok, lower=True)


	fields = [("ARTICLE", txt), ("SUMMARY", txt)]

	training, validation = data.TabularDataset.splits(path= "", train="training.csv",
	                                                  validation="validation.csv", format="csv", fields=fields)

	a = (training.examples[0])

	txt.build_vocab(training, vectors='glove.6B.300d')
	#sum.build_vocab(training, vectors='glove.6B.200d')
	tr_iter, val_iter = data.BucketIterator.splits((training,validation), batch_sizes=(128,128),
	                                               sort_key=lambda a: data.interleave_keys(len(a.ARTICLE), len(a.SUMMARY)), sort_within_batch=False,
	                                               repeat=False)

	#torch.save(txt, "article_field.pt")
	#torch.save(sum, "article_field.pt")
	#torch.save(training, "training.pt")
	#torch.save(validation, "validation.pt")
	#torch.save(tr_iter, "training_iterator.pt")
	#torch.save(val_iter, "validation_iterator.pt")
	return txt, tr_iter, val_iter


def expand(s, output):
	summaries = torch.zeros((output.shape[0], output.shape[1], output.shape[2]))

	for tok in range(output.shape[0]):
		for sent in range(output.shape[1]):
			idx = s[tok, sent]
			summaries[tok, sent, idx] = 1
	return summaries

def convert_to_text(out, vocab, pred=True):
	if pred:
		tok_matrix = torch.zeros((out.shape[0], out.shape[1]))
		for tok in range(out.shape[0]):
			for sent in range(out.shape[1]):
				#normalize first
				#norm_vec = out[tok, sent, :] / torch.sum(out[tok, sent, :])
				tok_matrix[tok, sent] = int(out[tok, sent, :].argmax())

				#tok_matrix[tok, sent] = int(out[tok, sent, :].argmax())
		#word_matrix = [[0 for x in range(out.shape[1])] for y in range(out.shape[0])]
		word_matrix = []
		for i in range(tok_matrix.shape[1]):
			word_matrix.append([])
			for j in range(tok_matrix.shape[0]):
				word_matrix[i].append(vocab.itos[int(tok_matrix[j,i])])

		return word_matrix
	else:
		word_matrix = []
		for i in range(out.shape[1]):
			word_matrix.append([])
			for j in range(out.shape[0]):
				word_matrix[i].append(vocab.itos[int(out[j, i])])
		return word_matrix

def detokenize(preds, labels):
	p = []
	l = []
	for i in range(len(preds)):
		sentence = ""
		for j in range(len(preds[i])):
			if preds[i][j][0] not in string.punctuation:
				sentence += " " + preds[i][j]
			else: sentence += preds[i][j]
		p.append(sentence)

	for i in range(len(labels)):
		sentence = ""
		for j in range(len(labels[i])):
			if labels[i][j][0] not in string.punctuation:
				sentence += " " + labels[i][j]
			else: sentence += labels[i][j]
		l.append(sentence)


	return p, l



get_data()
#txt, train, val = preprocess()
#batch_train = BatchPair(train)
#batch_val = BatchPair(val)