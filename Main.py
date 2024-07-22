import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision import transforms, datasets
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224,224)), # Size expected by pretrained ResNet Model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456,0.406], [0.229,0.224,0.225]) # Based on ResNet Model
])

# Preparing the dataset
def create_dataloaders(data_dir, transform, batch_size, num_workers):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    return dataloader

train_dir = 'data/train'
test_dir = 'data/test'
val_dir = 'data/val'

train_loader = create_dataloaders(train_dir, transform, batch_size=64, num_workers=8)
val_loader = create_dataloaders(val_dir, transform, batch_size=64, num_workers=8)
test_loader = create_dataloaders(test_dir, transform, batch_size=64, num_workers=8)

print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")
print(f"Number of test samples: {len(test_loader.dataset)}")

# The model is being loaded - ResNet18

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # Binary Classification (N or P)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # using Gpu if possible
model = model.to(device)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss/ len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1) # First variable is max value and 2nd is their indicies
            corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss/ len(dataloader.dataset)
    accuracy = corrects.double() / len(dataloader.dataset)
    return epoch_loss, accuracy

# Training the Model and saving the one with best validation scores
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
best_model_weights = None
best_accuracy = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

    if val_acc >= best_accuracy:
        best_accuracy = val_acc
        best_model_weights = model.state_dict().copy()

model.load_state_dict(best_model_weights)
torch.save(model.state_dict(), 'best_model.pth')

# Running the model on the test data
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}')

