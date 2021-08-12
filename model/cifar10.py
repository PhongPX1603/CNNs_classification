from torch import nn
import torch

class ConvBNReLU(nn.Module):
    '''Conv --> BN --> ReLU'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(self.bn)
        self.initialize()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x) 
        return x


class CIFAR10(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CIFAR10, self).__init__()
        self.conv1 = ConvBNReLU(in_channels=in_channels, out_channels=8, kernel_size= 3, stride=1, padding=1)
        self.conv2 = ConvBNReLU(in_channels=8, out_channels=16, kernel_size= 3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(in_channels=16, out_channels=32, kernel_size= 3, stride=1, padding=1)
        self.conv4 = ConvBNReLU(in_channels=32, out_channels=64, kernel_size= 3, stride=2, padding=1)
        self.conv5 = ConvBNReLU(in_channels=64, out_channels=128, kernel_size= 3, stride=1, padding=1)

        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # B, C, 1, 1
        self.fc1 = nn.Linear(in_features=128*8*8, out_features=128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=50)
        self.bn2 = nn.BatchNorm1d(num_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=num_classes)

        self.initialize()
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc3(x)

        return x