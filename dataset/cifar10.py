from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def Cifar10():
	mean = [0.4914, 0.4822, 0.4465]
	std = [0.247, 0.243, 0.261]

	transform = transforms.Compose([transforms.Resize(224),
									transforms.ToTensor(),
	                                transforms.Normalize(mean, std)])

	train_dataset = datasets.CIFAR10(root='dataset', train=True, transform=transform, download=True)
	valid_dataset = datasets.CIFAR10(root='dataset', train=False, transform=transform, download=True)

	train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers = 2)
	valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False, num_workers = 2)

	return train_loader, valid_loader
