import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

"""
Our Neural Network for evaluating Connect 4 game states.

This network classifies game board states into one of three categories: 
player winning (-1): tie (0): ai winning (1)

The architecture consists of four fully connected layers with ReLU activations in the first three layers and a final layer output 

This file trains our network and stores it to be loaded in efficiently
"""
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 34) # How many nodes in layer 1
        self.fc2 = nn.Linear(34, 26) # How many nodes in layer 1
        self.fc3 = nn.Linear(26, 18) # How many in layer 2 and so on
        self.fc4 = nn.Linear(18, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(x)
            _, outputs = scores.max(1)
            
            y = y.tolist()
            labels = [item[0] for item in y]

            outputs = outputs.tolist()

            for i in range(len(y)):
                if labels[i] == outputs[i]:
                    num_correct += 1

            num_samples += len(y)

    return num_correct / num_samples

if __name__ == '__main__':
    # Hyper parameters
    input_size = 42
    num_classes = 3
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.01

    # Loading data
    data = pd.read_csv('./data/c4_game_database.csv')

    features = data.columns[:-1].to_list()
    target = ['winner']

    x_data = data[features].to_numpy() # feature vectors
    y_data = data[target].to_numpy() # label values
    y_data = y_data + 1 # shifting y_data to coperate w/ 0-based index

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=0.30, random_state=42)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(input_size, num_classes).to(device)

    # Training the model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device).long().squeeze()
            
            # labels[labels == -9223372036854775808] = 0 <- uncomment on windows!!
            
            optimizer.zero_grad()

            # forward prop
            scores =  model(data)
            loss = loss_fn(scores, labels)

            # backward prop
            loss.backward()

            # descent
            optimizer.step()
        
    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}") # 95.51
    print(f"Accuracy on testing set: {check_accuracy(test_loader, model)*100:.2f}") # 95.37

    # Saving model
    torch.save(model.state_dict(), 'c4_model.pth')
