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

'''
In order to have 
'''
class AIEvaluator():
    def __init__(self):
        self.input_size = 42
        self.num_classes = 3
        self.batch_size = 64
        self.num_epochs = 5
        self.learning_rate = 0.01
        self.train_loader = None
        self.test_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NN(self.input_size, self.num_classes).to(self.device)

    def load_data(self):
        # Loading and formatting data
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
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def train(self):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for idx, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device).long().squeeze()

                optimizer.zero_grad()

                # forward prop
                scores = self.model(data)
                loss = loss_fn(scores, labels)

                # backward prop
                loss.backward()

                # descent
                optimizer.step()
    
    '''
    Main purpose of this model, which is to return a evaluation given a board state.
    This function will return a pos num if the AI is in a more favorable pos and negative if otherwise.
    '''
    def evaluate_pos(self, board):
        # Convert board list to tensor 
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        board_tensor = board_tensor.to(self.device)  # Send tensor to the correct device
        
        with torch.no_grad():
            score = self.model(board_tensor).tolist()[0]
            true_score = max(score)
            index = score.index(true_score) # Gets index which is the class the model put this board into
            
            # Labels: -1 0 1 correspond to Indices: 0 1 2
            if index == 2:
                return true_score 
            elif index == 0:
                return -1 * true_score # To keep in line with the minimax alg
            else:
                return 0


    def print_accuracy(self):
        print(f"Accuracy on training set: {check_accuracy(self.train_loader, self.model)*100:.2f}")
        print(f"Accuracy on testing set: {check_accuracy(self.test_loader, self.model)*100:.2f}")

# def driver():
#     eval = AIEvaluator()
#     eval.load_data()
#     eval.train()
#     eval.print_accuracy()
#     print(eval.evaluate_pos([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.0,0.0,-1.0,1.0,1.0,-1.0,-1.0,1.0,0.0,1.0,-1.0,-1.0,1.0,1.0,1.0,0.0,-1.0,-1.0,1.0,-1.0,1.0,-1.0,0.0,-1.0,-1.0,1.0,1.0,1.0,-1.0,0.0]))

# driver()