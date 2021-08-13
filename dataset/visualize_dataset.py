import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

def visualize(train_loader, args.device):
	train_iter = iter(train_loader)
	samples, labels = train_iter.next()

	classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 
	           5: 'dog',6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

	figure = plt.figure(figsize=(16, 16))

	rows = 8
	columns = 8

	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]

	mean = torch.tensor(mean, dtype=torch.float).reshape(1, 3, 1, 1).to(device)
	std = torch.tensor(std, dtype=torch.float).reshape(1, 3, 1, 1).to(device)

	samples = samples.to(device)
	labels = labels.to(device)

	samples = samples * std + mean  # B x C x H x W, torch.float
	samples = samples.mul(255.).to(torch.uint8)  # B x C x H x W, torch.uint8
	samples = samples.permute(0, 2, 3, 1)  # B x H x W x C, torch.uint8, RGB
	images = samples.cpu().numpy()
	images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images] # B x H x W x C, np.uint8, BGR

	labels = labels.cpu().numpy().tolist()

	for idx, (image, label) in enumerate(zip(images, labels)):
	    figure.add_subplot(rows, columns, idx + 1)
	    plt.imshow(image)
	    plt.axis("off")
	    plt.title(classes[label])

	plt.show()