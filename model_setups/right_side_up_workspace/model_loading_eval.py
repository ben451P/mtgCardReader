import torch.nn as nn
import torch, os
from torch.utils.data import TensorDataset, DataLoader

model_num = 3

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(BASE_DIR,"models",f"model{model_num}.pt")

dataset_path = "/Users/benlozzano/VS-Code-Coding/Ongoing/MTGCardReaderA2/right_side_up_workspace/dataset1.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    
model = CustomCNN(128, 4)
model.load_state_dict(torch.load(PATH, weights_only=True))

data = torch.load(dataset_path, map_location='cpu')
dataset = TensorDataset(data['images'], data['labels'])
test_dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

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



