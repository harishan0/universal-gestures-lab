import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from pprint import pprint

output_dim = 1  # binary classification for thumbs up or down
input_dim = 17  # 17 features
# TODO: Need to increase the number of inputs based on what I can distinguish between thumbs up and thumbs down
detect_threshold = 0.7  # threshold for classification as a thumbs up
# TODO: Might need to change this threshold depending on the losses and successes

SAVE_MODEL_PATH = "trained_model/"
SAVE_MODEL_FILENAME = "model_weights.json"

# Model
class TransformDataNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformDataNeuralNetModel, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        self.sigmoid = nn.Sigmoid()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function
        out = self.fc1(x)
        # Non-linearity
        out = self.sigmoid(out)
        # Linear function (readout)
        out = self.fc2(out)
        return torch.sigmoid(out)
    
        # TODO: Experiment with different activation functions to get the best accuracy
        # available activation functions include: relu

def main():
    train_path = "train_data/train_0.pt"
    test_path = "test_data/test_0.pt"
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    batch_size = 64
    n_iters = len(train_data) * 5  # 5 epochs
    num_epochs = int(n_iters / (len(train_data) / batch_size))

    X_train = torch.tensor(train_data[:, :-1])
    y_train = torch.tensor(train_data[:, -1])
    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), shuffle=True, batch_size=16
    )

    X_test = torch.tensor(test_data[:, :-1])
    y_test = torch.tensor(test_data[:, -1])
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), shuffle=True, batch_size=16
    )

    model = TransformDataNeuralNetModel(input_dim, 100, output_dim)
    # TODO: Might need to change the number of hidden neuron layers to more if I am adding more input features
    criterion = nn.BCELoss()
    learning_rate = 0.0004
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iter = 0

    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            Y = Y.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(X.float())
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()
            iter += 1

            if iter % 500 == 0:
                correct = 0
                total = 0
                for X, Y in test_loader:
                    outputs = model(X.float())
                    predicted = (outputs > detect_threshold).float()
                    total += Y.size(0)
                    correct += (predicted == Y.view(-1, 1)).sum().item()

                accuracy = 100 * correct / total
                print(
                    "Iteration: {}. Loss: {}. Accuracy: {}".format(
                        iter, loss.item(), accuracy
                    )
                )

    # Extract the model's state dictionary, convert to JSON serializable format
    state_dict = model.state_dict()
    serializable_state_dict = {key: value.tolist() for key, value in state_dict.items()}

    # Store state dictionary
    with open(SAVE_MODEL_PATH + SAVE_MODEL_FILENAME, "w") as f:
        json.dump(serializable_state_dict, f)

    print("\n--- Model Training Complete ---")
    print("\nModel weights saved to ", SAVE_MODEL_PATH + SAVE_MODEL_FILENAME)


if __name__ == "__main__":
    main()