import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np


import deeplake
ds = deeplake.load('hub://activeloop/11k-hands')

height, width = 800, 800
# Define a simple CNN model
class HandWaveCNN(nn.Module):
    def __init__(self):
        super(HandWaveCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (height // 8) * (width // 8), 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification (hand wave or not)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Instantiate the model
model = HandWaveCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


tform = transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.Resize(800),
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

#PyTorch Dataloader
# dataloader=ds.pytorch(batch_size=32, num_workers=2, transform={'images': tform, 'labels': None}, shuffle=True)
dataloader = ds.dataloader().transform({'images': tform, 'labels': None}).batch(32).shuffle().pytorch()

dataset_list = list(dataloader.dataset)

# Split the list into training and testing sets
train_size = int(0.8 * len(dataset_list))
test_size = len(dataset_list) - train_size

train_loader, test_loader = torch.utils.data.random_split(dataset_list, [train_size, test_size])


# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        accuracy = torch.sum(predicted == labels).item() / len(labels)
        print(f'Test accuracy: {accuracy}')

# Save the trained model for later use
torch.save(model.state_dict(), 'hand_waving_model.pth')
