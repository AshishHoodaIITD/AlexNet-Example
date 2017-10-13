import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

train_data = []
train_label = []
file = open("RNN_Data_files/train_sentences.txt", "r")
for line in file:
	train_data.append(line.split())
file = open("RNN_Data_files/train_tags.txt", "r")
for line in file:
	train_label.append(line.split())

test_data = []
test_label = []
file = open("RNN_Data_files/val_sentences.txt", "r")
for line in file:
	test_data.append(line.split())
file = open("RNN_Data_files/val_tags.txt", "r")
for line in file:
	test_label.append(line.split())

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

class tagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(tagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# Refer to the Pytorch documentation to see exactly
		# why they have this dimensionality.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
				autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		# print(embeds)
		# print(embeds.view(1,len(sentence), -1))
		# print(self.hidden)
		self.hidden = self.lstm(embeds.view(len(sentence),1, -1), self.hidden)
		tag_space = self.hidden2tag(self.hidden[0].view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores
EMBEDDING_DIM=64
HIDDEN_DIM = 64
model = tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)
# print(tag_scores)

for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
	for i in range(len(train_data)):
		sentence = train_data[i]
		tags = train_label[i]
		model.zero_grad()

		# Also, we need to clear out the hidden state of the LSTM,
		# detaching it from its history on the last instance.
		model.hidden = model.init_hidden()

		# Step 2. Get our inputs ready for the network, that is, turn them into
		# Variables of word indices.
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets = prepare_sequence(tags, tag_to_ix)

		# Step 3. Run our forward pass.
		tag_scores = model(sentence_in)

		# Step 4. Compute the loss, gradients, and update the parameters by
		#  calling optimizer.step()
		loss = loss_function(tag_scores, targets)
		loss.backward()
		optimizer.step()
	acc = test(model)
	print('Accuracy of the network on the validation set: %d %%' % (
		acc))

# # See what the scores are after training
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
# tag_scores = model(inputs)
# # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
# #  for word i. The predicted tag is the maximum scoring tag.
# # Here, we can see the predicted sequence below is 0 1 2 0 1
# # since 0 is index of the maximum value of row 1,
# # 1 is the index of maximum value of row 2, etc.
# # Which is DET NOUN VERB DET NOUN, the correct sequence!
# print(tag_scores)


def test(model):
	total =0
	correct =0
	for i in range(len(test_data)):
		input_data = prepare_sequence(test_data[i], word_to_ix)
		input_tag = prepare_sequence(test_label[i], tag_to_ix)
		tag_scores = model(input_data)
		_, predicted = torch.max(tag_scores, 1)
		total+= len(input_tag)
		correct += (predicted == input_tag).sum()
	return (100*correct/total)





