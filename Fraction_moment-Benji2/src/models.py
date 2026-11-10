import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calcul dynamique de la taille de la couche linéaire
        # Pour MNIST (28x28) -> après 2 poolings -> 7x7
        # Pour CIFAR10 (32x32) -> après 2 poolings -> 8x8
        dummy_size = 28 if in_channels == 1 else 32
        out_size = dummy_size // 4 
        
        self.fc1 = nn.Linear(32 * out_size * out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # Utilisation de x.size(0) pour être robuste
        x = x.view(x.size(0), -1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x