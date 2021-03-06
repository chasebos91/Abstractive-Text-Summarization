import torch
import time
from data_util import preprocess
import vanillaseq2seq as vs2s
import pickle


best = float("inf")
text, train_iter, val_iter = preprocess()
model = vs2s.setup(text)
optim = torch.optim.SGD(model.parameters(), .01)
train_loss_list = []
val_loss_list_long = []

for e in range(150):
	start = time.time()
	model.tf = True
	train_loss, loss_list = vs2s.train(model, train_iter, optim)
	model.tf = False
	val_loss, val_loss_list, rouge_scores = vs2s.eval(model, val_iter)
	train_loss_list.append(loss_list)
	val_loss_list_long.append(val_loss_list)
	end = time.time()
	t = vs2s.keep_time(start, end)
	if val_loss < best:

		fr = open("vanilla_rouge.pk", "wb")
		torch.save(model.state_dict(), "vanillaseq2seq.pt")
		pickle.dump(rouge_scores, fr)
		fr.close()
		fl = open("vanilla_loss_small.pk", "wb")
		pickle.dump(loss_list, fl)
		fl.close()


	print("Time: ", str(t))
	print("train loss: ", train_loss)
	print("validation loss: ", val_loss)

f = open("vanilla_loss.pk", "wb")
fv = open("vanilla_val_loss.pk", "wb")
pickle.dump(train_loss_list, f)
pickle.dump(val_loss_list_long, fv)
f.close()
fv.close()