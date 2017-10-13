import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.autograd import Variable

start_time = time.time()

#reading training data
train_data = []
train_label = []

file = open("RNN_Data_files/train_sentences.txt", "r")
for line in file:
	train_data.append(line.split())

file = open("RNN_Data_files/train_tags.txt", "r")
for line in file:
	train_label.append(line.split())

#reading validation data
val_data = []
val_label = []
file = open("RNN_Data_files/val_sentences.txt", "r")
for line in file:
	val_data.append(line.split())
file = open("RNN_Data_files/val_tags.txt", "r")
for line in file:
	val_label.append(line.split())

word_to_ix = {}
tag_to_ix = {}
for sent in train_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sent in val_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for tags in train_label:
	for word in tags:
		if word not in tag_to_ix:
			tag_to_ix[word] = len(tag_to_ix)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return tensor.cuda()

train_data1 =[]
train_label1 =[]
for i in range(len(train_data)):
	train_data1.append(prepare_sequence(train_data[i], word_to_ix))
	train_label1.append(prepare_sequence(train_label[i], tag_to_ix))
train_data = train_data1
train_label = train_label1

val_data1 =[]
val_label1 =[]
for i in range(len(val_data)):
	val_data1.append(prepare_sequence(val_data[i], word_to_ix))
	val_label1.append(prepare_sequence(val_label[i], tag_to_ix))
val_data = val_data1
val_label = val_label1


# tmp =[]
# for i in train_data:
# 	tmp.append(len(i))
# train_data = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(train_data,tmp),batch_first=True)
# print(hello)
# train_data = Variable(torch.LongTensor(train_data))
# train_label = Variable(torch.LongTensor(train_label))
# val_data = Variable(torch.LongTensor(val_data))
# val_label = Variable(torch.LongTensor(val_label))

# print('time elapsed in preprocessing ='+str(time.time()-start_time))

# transformed_dataset =[]
# for i in range(len(train_data)):
# 	transformed_dataset.append({'data':train_data[i],'label':train_label[i]})
# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]

#     print(i, sample['data'].size(), sample['label'].size())

#     if i == 3:
#         break

# dataloader = torch.utils.data.DataLoader(transformed_dataset, batch_size=4,shuffle=True, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['data'].size(),
#           sample_batched['label'].size())
# print('time elapsed in preprocessing ='+str(time.time()-start_time))


class tagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(tagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.RNNCell(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return autograd.Variable(torch.zeros( 1, self.hidden_dim).cuda())

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		output = []
		for word in embeds:
			self.hidden = self.lstm(word.view(1, -1), self.hidden)
			output.append(self.hidden)
		output = torch.stack(output)
		tag_space = self.hidden2tag(output.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores

EMBEDDING_DIM = 32
HIDDEN_DIM = 64
model = tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)
# print(tag_scores)

def validation(model):
	total = 0
	correct = 0
	for i in range(len(val_data)):
		input_data = Variable(val_data[i])
		input_tag = val_label[i]
		tag_scores = model(input_data)
		_, predicted = torch.max(tag_scores.data, 1)
		total+= len(input_tag)
		correct += (predicted == input_tag).sum()
	return (100*(float(correct)/float(total)))

for epoch in range(3):  # again, normally you would NOT do 300 epochs, it is toy data
	for i in range(len(train_data)):
		sentence = Variable(train_data[i])
		tags = Variable(train_label[i])
		model.zero_grad()

		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		model.hidden = model.init_hidden()

		# Step 3. Run our forward pass.
		tag_scores = model(sentence)

		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		loss = loss_function(tag_scores, tags)
		loss.backward()
		optimizer.step()
		# if(i%100==0):
		# 	print(i)
	acc = validation(model)
	print('Accuracy of the network on the validation set: %d %%' % (
		acc))








