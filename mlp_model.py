
import torch.nn as nn

class mlp_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 2nd hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # output layer
        self.softmax = nn.Softmax(dim=1)
        # self.dropout1 = nn.Dropout(0.2)


    def forward(self, x):

        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)

        return output
