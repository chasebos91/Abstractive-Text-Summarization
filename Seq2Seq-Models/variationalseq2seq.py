import torch
from torch import nn
from data_util import convert_to_text, detokenize
import time

import numpy as np
from rouge import Rouge
import random

# TODO save model parameters after training

#a variational BiLSTM seq2seq


class variationalseq2seq(nn.Module):
	def __init__(self, input_size, emb_size, hid_size, out_size, text):
		super(variationalseq2seq, self).__init__()
		self.input_size = input_size
		self.emb_size = emb_size
		self.hid_size = hid_size
		self.out_size = out_size
		self.text = text
		self.tf = None
		self.encoder = nn.ModuleDict([["embedding", nn.Embedding(input_size, emb_size)],
		                ["bilstm", nn.LSTM(emb_size, hid_size, 2, bidirectional=True)],
		                ["mu", nn.Linear(hid_size, hid_size)],
		                ["logvar", nn.Linear(hid_size, hid_size)]])
		self.encoder["embedding"].weight.data.copy_(text.vocab.vectors)
		self.decoder =  nn.ModuleDict([["hid", nn.Linear(hid_size, hid_size)],
		                               ["embedding", nn.Embedding(out_size, emb_size)],
		                ["bilstm", nn.LSTM(emb_size, hid_size, 2, bidirectional=True)],
		                ["predict", nn.Linear(hid_size*2, out_size)],
						["softmax", nn.LogSoftmax(1)]])
		self.decoder["embedding"].weight.data.copy_(text.vocab.vectors)

		self.crit = nn.NLLLoss(ignore_index = text.vocab.stoi['<pad>'])
		self.max_len = 15
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def forward(self, article, summary):
		outs = torch.zeros(summary.shape[0], summary.shape[1], self.out_size)
		emb = self.encoder["embedding"](article)
		x, (hid, c) = self.encoder["bilstm"](emb)
		#hid.view(article.shape[0], self.hid_size * 2)
		#reparameterization trick
		mu = self.encoder["mu"](hid)
		logvar = self.encoder["logvar"](hid)
		stddev = torch.exp(.5 * logvar)
		z = torch.randn([article.shape[1], self.hid_size])
		if torch.cuda.is_available():
			z = z.cuda()
		z = z * stddev + mu
		dec_in = summary[0, :]
		h = self.decoder["hid"](z)
		for i in range(0, summary.shape[0]):
			tf = random.random() < .5
			# reshape h maybe
			#h = h.view(article.shape[1], 4, self.hid_size*2)
			dec_in = dec_in.unsqueeze(0)
			x = self.decoder["embedding"](dec_in)
			x, (h, c) = self.decoder["bilstm"](x, (h, c))
			x = x.squeeze(0)
			x = self.decoder["predict"](x)
			#prob = self.decoder["softmax"](x)
			p = self.decoder["softmax"](x)
			outs[i] = p
			#outs[i] = x
			#dec_in = p.argmax(1)
			if self.tf:
				if tf: dec_in = summary[i]
				else: dec_in = p.argmax(1)
			else: dec_in = p.argmax(1)


		return outs, mu, logvar, z

	def KL_loss(self, prob, mu, logvar, summary, i):
		nll = self.crit(prob, summary)
		KLl = -.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
		# mitigate KL vanishing
		annealing_term = float(1/(1 + np.exp(-0.0025*(i-2500))))
		return nll, KLl, annealing_term

def setup(text):
	pad_id = text.vocab.stoi["<pad>"]

	unk_id = text.vocab.stoi[text.unk_token]



	model = variationalseq2seq(len(text.vocab), 300, 1024, len(text.vocab), text)
	model.to(model.device)
	model.encoder["embedding"].weight.data[unk_id] = torch.zeros(300)
	model.encoder["embedding"].weight.data[pad_id] = torch.zeros(300)
	model.encoder["embedding"].weight.requires_grad = False

	for n, p in model.named_parameters():
		nn.init.uniform_(p.data, -.05, .05)

	return model

def train(model, iter, opt):
	model.train()
	total_loss = 0
	loss_list = []
	step =0
	for i, b in enumerate(iter):
		a = b.ARTICLE
		s = b.SUMMARY
		st = time.time()
		opt.zero_grad()
		out, mu, logvar, z = model(a, s)
		out = out[1:].view(-1, out.shape[-1])
		s = s[1:].view(-1)
		NLL, KL, w = model.KL_loss(out.float(), mu, logvar, s, step)
		loss = NLL + w * KL
		loss.backward()
		opt.step()
		total_loss += loss.item()
		print(loss.item())
		step += 1
		print(i, loss.item())
		end = time.time()
		print("time elapsed: ", keep_time(st, end))

		if i % 100 == 0:
			loss_list.append(loss.item())
			#torch.save(model.state_dict(), "variationalseq2seq.pt")
		#if i > 1: return total_loss / len(iter), loss_list

	return total_loss/ len(iter), loss_list

def eval(model, iter):

	model.eval()
	total_loss = 0
	rouge_scores  = []
	loss_list = []

	r = Rouge()
	step = 0



	with torch.no_grad():

		for i, b in enumerate(iter):
			a = b.ARTICLE
			s = b.SUMMARY
			output, mu, logvar, z = model(a, s)
			out = output[1:].view(-1, output.shape[-1])
			summary = s[1:].view(-1)
			NLL, KL, w = model.KL_loss(out.float(), mu, logvar, summary, step)
			step += 1
			loss = NLL + w * KL
			total_loss += loss.item()

			if i % 10 == 0:
				# print("Loss at iteration " + str(i) +  ": ", loss.item())
				loss_list.append(loss.item())

			"""
			if i > 1:
				preds = convert_to_text(output, model.text.vocab)
				labels = convert_to_text(s, model.text.vocab, False)
				preds, labels = detokenize(preds, labels)

				for i in range(len(preds)):
					rouge_scores.append([i, preds[i], labels[i], r.get_scores(preds[i], labels[i])])

				return total_loss / len(iter), loss_list, rouge_scores
			"""
		preds = convert_to_text(output, model.text.vocab)
		labels = convert_to_text(s, model.text.vocab, False)
		preds, labels = detokenize(preds, labels)


		for i in range(len(preds)):
			rouge_scores.append([i, preds[i], labels[i], r.get_scores(preds[i], labels[i])])


	return total_loss/len(iter), loss_list, rouge_scores

def keep_time(start, end):
	return (end - start) / 60


#MAIN
