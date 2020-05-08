import torch
from torch import nn
import time
from data_util import preprocess, convert_to_text, detokenize

from rouge import Rouge
import random

#a BiLSTM seq2seq


class seq2seq(nn.Module):
	def __init__(self, input_size, emb_size, hid_size, out_size, text):
		super(seq2seq, self).__init__()
		self.encoder = encoder(input_size, emb_size, hid_size, text)
		self.decoder = decoder(emb_size, hid_size, out_size, text)
		# ignore_index = text.vocab.stoi['<pad>']
		self.crit = nn.NLLLoss()
		self.max_len = 15
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.text = text
		self.tf = True


	def forward(self, article, summary):

		outs = torch.zeros(summary.shape[0], summary.shape[1], self.decoder.out_size).to(self.device)
		h, c = self.encoder(article)

		dec_in = summary[0, :]

		for i in range(1, summary.shape[0]):
			tf = random.random() < .5
			out, h, c = self.decoder(dec_in, h, c)
			outs[i] = out
			#teacher forcing in training
			if self.tf:
				if tf: dec_in = summary[i]
				else: dec_in = out.argmax(1)
			else: dec_in = out.argmax(1)

		return outs


class encoder(nn.Module):
	def __init__(self, input_size, emb_size, hid_size, text):
		super(encoder, self).__init__()
		self.input_size = input_size
		self.hid_size = hid_size
		self.embedding = nn.Embedding(input_size, emb_size)
		self.embedding.weight.data.copy_(text.vocab.vectors)
		self.bilstm = nn.LSTM(emb_size, hid_size, 2, bidirectional=True, dropout=.5)
		self.dropout = nn.Dropout(.5)

	def forward(self, x):
		embed = self.embedding(x)
		out, (h, c) = self.bilstm(embed)
		return h, c

	def init_hidden_state(self):
		return torch.zeros(1, 1, self.hid_size)

class decoder(nn.Module):
	def __init__(self, emb_size, hid_size, out_size, text):
		super(decoder, self).__init__()
		self.out_size = out_size
		self.hid_size = hid_size
		self.attn1 = nn.Linear(hid_size * 2, 10)
		self.attn2 = nn.Linear(hid_size, 1, bias=False)
		self.embedding = nn.Embedding(out_size, emb_size)
		self.embedding.weight.data.copy_(text.vocab.vectors)
		self.bilstm = nn.LSTM(emb_size, hid_size, 2, bidirectional=True, dropout=.5)
		self.linear = nn.Linear(hid_size * 2,out_size)
		self.dropout = nn.Dropout(.5)
		self.softmax = nn.LogSoftmax(1)

	def forward(self, x, h, c):
		x = x.unsqueeze(0)
		x = self.embedding(x)
		x = self.dropout(x)
		x, (h, c) = self.bilstm(x, (h, c))
		x = x.squeeze(0)
		x = self.linear(x)
		x = self.softmax(x)
		return x, h, c


	def init_hidden_state(self):
		return torch.zeros(1, 1, self.hid_size)


def setup(text):
	pad_id = text.vocab.stoi["<pad>"]

	unk_id = text.vocab.stoi[text.unk_token]

	model = seq2seq(len(text.vocab), 300, 1024, len(text.vocab), text)
	model.to(model.device)
	model.encoder.embedding.weight.data[unk_id] = torch.zeros(300)
	model.encoder.embedding.weight.data[pad_id] = torch.zeros(300)
	model.encoder.embedding.weight.requires_grad = False

	for n, p in model.named_parameters():
		nn.init.uniform_(p.data, -.05, .05)

	return model

def train(model, iter, opt):
	model.train()
	total_loss = 0
	loss_list = []

	for i, b in enumerate(iter):
		a = b.ARTICLE
		s = b.SUMMARY
		st = time.time()
		opt.zero_grad()
		out = model(a, s)
		output = out[1:].view(-1, out.shape[-1])
		#summary = expand(s, out)
		summary = s[1:].view(-1)
		#summary = summary[1:].view(-1, summary.shape[-1])
		loss = model.crit(output, summary)
		loss.backward()
		opt.step()
		total_loss += loss.item()
		print(i, loss.item())
		end = time.time()
		print("time elapsed: ", keep_time(st, end))
		if i % 100 == 0:
			print("Loss at iteration " + str(i) +  ": ", loss.item())
			loss_list.append(loss.item())



	return total_loss/ len(iter), loss_list




def eval(model, iter):

	model.eval()
	total_loss = 0
	rouge_scores = []
	r = Rouge()
	loss_list = []

	with torch.no_grad():

		for i, b in enumerate(iter):
			a = b.ARTICLE
			s = b.SUMMARY
			out = model(a, s)
			output = out[1:].view(-1, out.shape[-1])
			summary = s[1:].view(-1)
			loss = model.crit(output, summary)
			total_loss += loss.item()


			if i % 10 == 0:
				loss_list.append(loss.item())



		preds = convert_to_text(out, model.text.vocab)
		labels = convert_to_text(s, model.text.vocab, False)
		preds, labels = detokenize(preds, labels)

		for i in range(len(preds)):
			rouge_scores.append([i, preds[i], labels[i], r.get_scores(preds[i], labels[i])])

	return total_loss/len(iter), loss_list, rouge_scores



def keep_time(start, end):
	return (end - start) / 60

#MAIN




