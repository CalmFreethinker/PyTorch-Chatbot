'''running the training data through neural network'''
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # tokenizing all words in the patterns (potential questions)
        all_words.extend(w)  # we use extend instead of append because pattern is already an array and we don't want to put an array in an array
        xy.append((w, tag))
    
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words] #stemming all the words and ignoring punctuation 
all_words = sorted(set(all_words))  # using set() removes duplicate elements
tags = sorted(set(tags))  #adds unique labels (not necessary but do it in to be safe) 


X_train = []
y_train = []
#Getting the token and tag
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words) 
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #cross entropy loss
    
X_train = np.array(X_train)
y_train = np.array(y_train)

#for debugging purposes
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #Go through the dataset
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    #check the length of the dataset
    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 100

dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
    
#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
