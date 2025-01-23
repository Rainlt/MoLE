import torch
import json

import torch.nn as nn
import torch.optim as optim
import pdb
from sklearn import svm
# Define the linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        return out



# Load the data from 'token_list.json'
with open('token_list_1000.json', 'r') as f:
    data = json.load(f)

# Prepare the data
inputs = torch.tensor([item[0] for item in data])
labels = torch.tensor([int(item[2]) for item in data])

#取出labels中为0的label，以及对应的inputs，同时采样相同数量的label为1的label和对应的inputs

zero_labels = labels[labels == 0]
zero_inputs = inputs[labels == 0]
one_labels = labels[labels == 1]
one_inputs = inputs[labels == 1]
sampled_one_labels = one_labels[:len(zero_labels)]
sampled_one_inputs = one_inputs[:len(zero_labels)]
inputs = torch.cat((zero_inputs, sampled_one_inputs), 0)
labels = torch.cat((zero_labels, sampled_one_labels), 0)
#打乱labels的顺序，同时打乱inputs的顺序
shuffle_indices = torch.randperm(len(labels))
inputs = inputs[shuffle_indices]
labels = labels[shuffle_indices]
#按8:2划分训练集与验证集
train_size = int(0.8 * len(labels))
train_inputs = inputs[:train_size]
train_labels = labels[:train_size]
test_inputs = inputs[train_size:]
test_labels = labels[train_size:]


# Define the model
input_dim = 32
hidden_dim=16
output_dim = 2
model = LinearModel(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model

# Convert tensors to numpy arrays
train_inputs_np = train_inputs.numpy()
train_labels_np = train_labels.numpy()

# Create and train the SVM classifier
svm_classifier = svm.SVC()

svm_classifier.fit(train_inputs_np, train_labels_np)

# Test the modelel
test_inputs_np = test_inputs.numpy()
test_labels_np = test_labels.numpy()
predicted = svm_classifier.predict(test_inputs_np)

print("predicted",predicted)
print("test_labels_np",test_labels_np)
# Calculate accuracy
correct = (predicted == test_labels_np).sum().item()
accuracy = correct / len(test_labels_np) * 100
print(f'Accuracy: {accuracy:.2f}%')

