import torch
import variationalseq2seq
import pickle
from data_util import preprocess

text, train_iter, val_iter = preprocess()
model = variationalseq2seq.setup(text)
params = torch.load("variationalseq2seq.pt")
model.load_state_dict(params)
for p in model.parameters():
	p.requires_grad = False

model.eval()
optim = torch.optim.SGD(model.parameters(), .01)
train_loss_list = []
val_loss_list_long = []

for e in range(1):
	model.tf = False
	val_loss, val_loss_list, rouge_scores = variationalseq2seq.eval(model, val_iter)

	val_loss_list_long.append(val_loss_list)
	fr = open("variational_rouge.pk", "wb")
	pickle.dump(rouge_scores, fr)
	print("validation loss: ", val_loss)

f = open("variational_loss.pk", "wb")
fv = open("variational_val_loss.pk", "wb")
pickle.dump(train_loss_list, f)
pickle.dump(val_loss_list_long, fv)
f.close()
fv.close()