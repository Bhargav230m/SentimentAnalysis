# An extremely simple model
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.silu(self.fc1(x)) # [batch_size, hidden_size]
        x = F.silu(self.fc2(x)) # [hidden_size, hidden_size]
        x = self.fc3(x) # [hidden_size, output_size]

        return F.log_softmax(x, dim=1)