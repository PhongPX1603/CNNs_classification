from torch import nn
import torch


class DeepwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DeepwiseConv, self).__init__()
        self.dw_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                              out_channels=in_channels,
                                              groups=in_channels,
                                              kernel_size=3,
                                              stride=stride,
                                              padding=1),
                                    nn.BatchNorm2d(num_features=in_channels),
                                    nn.ReLU6(inplace=True))
        
        self.pw_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=1,
                                               stride=1
                                               ),
                                     nn.BatchNorm2d(num_features=out_channels),
                                     nn.ReLU6(inplace=True))

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x        
        
class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU6(inplace=True)

        self.block1 = DeepwiseConv(in_channels=32, out_channels=64, stride=1)
        self.block2 = DeepwiseConv(in_channels=64, out_channels=128, stride=2)
        self.block3 = DeepwiseConv(in_channels=128, out_channels=128, stride=1)
        self.block4 = DeepwiseConv(in_channels=128, out_channels=256, stride=2)
        self.block5 = DeepwiseConv(in_channels=256, out_channels=256, stride=1)
        self.block6 = DeepwiseConv(in_channels=256, out_channels=512, stride=2)
        self.block_x5 = DeepwiseConv(in_channels=512, out_channels=512, stride=1)
        self.block7 = DeepwiseConv(in_channels=512, out_channels=1024, stride=2)
        self.block8 = DeepwiseConv(in_channels=1024, out_channels=1024, stride=1)
 
        self.avg = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)   
        x = self.block3(x)   
        x = self.block4(x)   
        x = self.block5(x)   
        x = self.block6(x) 

        for _ in range(5):  
            x = self.block_x5(x)

        x = self.block7(x)
        x = self.block8(x)              
        x = self.avg(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x