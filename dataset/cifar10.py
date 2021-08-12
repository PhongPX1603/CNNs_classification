from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def Cifar10():
	mean = [0.4914005398750305, 0.4821619391441345, 0.44653016328811646]
	std = [0.1952536255121231, 0.19247283041477203, 0.1941995918750763]

	transform = transforms.Compose([transforms.ToTensor(),
	                                transforms.Normalize(mean, std)])

	train_dataset = datasets.CIFAR10(root='dataset', train=True, transform=transform, download=True)
	valid_dataset = datasets.CIFAR10(root='dataset', train=False, transform=transform, download=True)

	train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

	return train_loader, valid_loader
