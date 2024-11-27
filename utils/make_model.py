import torch.nn as nn
from torchvision import transforms, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def create_resnet_model(num_classes):
    """Creates a ResNet model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_vit_model(num_classes):
    """Creates a Vision Transformer (ViT) model."""
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

# Custom CNN Model with additional layers
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: (3, H, W) -> Output: (16, H, W)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: (32, H/2, W/2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: (64, H/4, W/4)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: (128, H/8, W/8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves H, W

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 14 * 14, 256)  # Adjusted for (224, 224) input size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Activation function and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = self.pool(self.relu(self.conv3(x)))  # Conv3 + ReLU + Pool
        x = self.pool(self.relu(self.conv4(x)))  # Conv4 + ReLU + Pool

        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))  # Fully Connected 1 + ReLU
        x = self.dropout(x)  # Dropout for regularization
        x = self.relu(self.fc2(x))  # Fully Connected 2 + ReLU
        x = self.fc3(x)  # Fully Connected 3 (Output layer)
        return x
