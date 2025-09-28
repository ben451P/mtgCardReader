import torch
import torch.nn as nn
import os, random
import cv2
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

base = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset"
dataset = []
for path in os.listdir(base):
    for _ in range(10):
        img = cv2.imread(os.path.join(base,path))
        dataset.append(img)

labels = []
for i in range(len(dataset)):
    choice = random.randint(0,3)
    for _ in range(choice):
        dataset[i] = cv2.rotate(dataset[i],cv2.ROTATE_90_CLOCKWISE)
    dataset[i] = cv2.resize(dataset[i], (256,256))
    labels.append(choice)
trainX, testX, trainy, testy = train_test_split(dataset,labels,train_size=.8,random_state=0)

train_dataset = TensorDataset(torch.tensor(trainX, dtype=torch.float32),torch.tensor(trainy, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(testX, dtype=torch.float32),torch.tensor(testy, dtype=torch.long))

train_dataloader = DataLoader(train_dataset,batch_size=16)
test_dataloader = DataLoader(test_dataset,batch_size=16)

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
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# class CustomCNN(nn.Module):
#     def __init__(self, hidden_nodes, output_nodes, conv_kernel_size=3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, conv_kernel_size, padding=1)
#         self.conv2 = nn.Conv2d(64, hidden_nodes, conv_kernel_size, padding=1)
#         self.pooling = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
#         self.fc1 = nn.Linear(hidden_nodes * 8 * 8, 16)
#         self.fc2 = nn.Linear(16, output_nodes)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.pooling(x)
#         x = self.relu(self.conv2(x))
#         x = self.pooling(x)
#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

input_features = trainX[0]

model = CustomCNN(128,4)
# model = CustomCNN(128,4)

optimizer = Adam(model.parameters(),lr=.0001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

model.train()
for epoch in range(epochs):
    loss1 = 0
    for batch in train_dataloader:
            y = model(batch[0].permute(0, 3, 1, 2))
            loss = loss_function(y,batch[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss1 += loss.item()
    print(epoch, loss1/len(train_dataloader))

model.eval()
with torch.no_grad():
    for inputs, labels in test_dataloader:
        y = model(inputs.permute(0, 3, 1, 2))
        # _, predicted = torch.max(outputs, 1)
        # correct += (predicted == y).sum().item()
        # total += y.size(0)

        _, predicted = torch.max(y.data, 1)
        correct_predictions = (predicted == labels).sum().item()
        print(correct_predictions / y.size(0))