'''feedforward neural network'''
import torch
import torch.nn as nn

# creating the neural network that will be used to train the data
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        #activation function
        self.relu = nn.ReLU()
    #this is the neural network
    def forward(self, x):
        out = self.l1(x)
        #passing each input through relu to keep values positive
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out

