# Use this to make prediction on new images individually

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import read_image
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

# Define the transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Same as before
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

def predict_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Map prediction to class labels
    class_names = ['normal', 'pneumonia']
    return class_names[predicted.item()]

# Insert Image Here
image_path = 'pneumonia.jpeg'
prediction = predict_image(image_path, model, transform, device)
print(f'Predicted Class: {prediction}\nActual Class: {image_path.split(".")[0]}')
