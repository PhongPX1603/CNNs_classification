import torch
import torch.nn as nn


class SEblock(nn.Module):
	def __init__(self, num_channels, ratio_reduce):
		super(SEblock, self).__init__()
		num_channels_reduce = num_channels // ratio_reduce
		self.ratio_reduce = ratio_reduce
		self.fc1 = nn.Linear(num_channels, num_channels_reduce, bias=True)
		self.fc2 = nn.Linear(num_channels_reduce, num_channels, bias=True)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor):
		batch_size, num_channels, H, W = input_tensor.size()

		squeeze = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

		x = self.fc1(squeeze)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)

		B, C = squeeze.size()

		output_tensor = torch.mul(input_tensor, x.view(B, C, 1, 1))

		return output_tensor

class block(nn.Module):
	def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, ratio_reduce=2):
		super(block, self).__init__()
		self.expansion = 4
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()

		#SE
		self.seblock = SEblock(out_channels, ratio_reduce=ratio_reduce)
		self.identity_downsample = identity_downsample

	def forward(self, x):
		identity = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.seblock(x)

		if self.identity_downsample is not None:
			identity = self.identity_downsample(identity)

		x += identity
		x = self.relu(x)

		return x
		
class Resnet(nn.Module):
	def __init__(self, block, layers, image_channels, num_classes):
		super(Resnet, self).__init__()
		self.in_channels = 64
		self.conv1 = nn.Conv2d(image_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		#Resnet
		self.layer1 = self.make_layer(block, num_residual_block=layers[0], out_channels=64, stride=1)
		self.layer2 = self.make_layer(block, num_residual_block=layers[1], out_channels=128, stride=2)
		self.layer3 = self.make_layer(block, num_residual_block=layers[2], out_channels=256, stride=2)
		self.layer4 = self.make_layer(block, num_residual_block=layers[3], out_channels=512, stride=2)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)

		return x

	def make_layer(self, block, num_residual_block, out_channels, stride):
		identity_downsample = None
		layers = []

		if stride != 1:
			identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, 
								      out_channels, 
								      kernel_size=1, 
								      stride=stride),
  							    nn.BatchNorm2d(out_channels))

		layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
		self.in_channels = out_channels

		for i in range(num_residual_block - 1):
			layers.append(block(self.in_channels, out_channels))

		return nn.Sequential(*layers)
