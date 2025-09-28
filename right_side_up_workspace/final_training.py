import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

dataset_path = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset1.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.load(dataset_path, map_location='cpu')
dataset = TensorDataset(data['images'], data['labels'])

# total = len(dataset)
train_size = .8

train_dataset, test_dataset = random_split(dataset,[train_size,1-train_size])

train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=True)

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

model = CustomCNN(128,4)

optimizer = Adam(model.parameters(),lr=.0001)
loss_function = nn.CrossEntropyLoss()
epochs = 20

loss_hist = []

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs.permute(0, 3, 1, 2))
        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(train_dataset)
    loss_hist.append(avg_loss)
    print(f"Epoch {epoch}: training loss = {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.permute(0, 3, 1, 2))
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"  Test accuracy: {acc:.4f}")

torch.save(model.state_dict(),"models/model3.pt")

plt.plot(loss_hist)
plt.show()