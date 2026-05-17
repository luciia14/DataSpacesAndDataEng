import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First Convolutional Block: 64x64 -> 32x32
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # Second Convolutional Block: 32x32 -> 16x16
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Third Convolutional Block: 16x16 -> 8x8
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input size is 64 channels * 8 height * 8 width
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x