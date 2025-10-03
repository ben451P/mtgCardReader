import torch.nn as nn
from torch import flatten

class CustomCNN(nn.Module):
    def __init__(self,hidden_nodes,output_nodes,conv_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,conv_kernel_size,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,hidden_nodes,conv_kernel_size,stride=1,padding=1)
        self.fc1 = nn.Linear(hidden_nodes * 32 * 32,16)
        self.fc2 = nn.Linear(16,output_nodes)
        self.pooling = nn.MaxPool2d((2,2),stride=2)
        self.adaPool = nn.AdaptiveMaxPool2d((32,32))
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pooling(x)
        x = self.relu(self.conv2(x))
        x = self.pooling(x)
        x = self.adaPool(x)
        x = flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x