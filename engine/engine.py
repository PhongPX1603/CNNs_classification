from torch import nn
import torch

def train_epoch(data_loader, model, criterion, optimizer, device='cuda'):
	model.to(device)
	model.train()

	accuracies, losses = [], []

	for samples, targets in data_loader:
		samples = samples.to(device)
		targets = targets.to(device)

		preds = model(samples)
		loss = criterion(targets, preds)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		preds = nn.Softmax(dim=1)(preds)
		preds = preds.argmax(dim=1)

		correct = torch.sum(preds == targets)
		accuracy = correct / targets.shape[0]

		accuracies.append(accuracy)
		losses.append(loss)

	average_loss = sum(losses) / len(losses)
	average_accuracies = sum(accuracies) / len(accuracies)

	return average_accuracies, average_loss

def valid_epoch(data_loader, model, criterion, device='cuda'):
	model.to(device)
	model.eval()

	accuracies, losses = [], []

	for samples, targets in data_loader:
		samples = samples.to(device)
		targets = targets.to(device)

		preds = model(samples)
		loss = criterion(preds, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		preds = nn.Softmax(dim=1)(preds)
		preds = preds.argmax(dim=1)
		correct = torch.sum(preds == targets)
		accuracy = correct / targets.shape[0]

		accuracies.append(accuracy)
		losses.append(loss)

	average_loss = sum(losses) / len(losses)
	average_accuracies = sum(accuracies) / len(accuracies)

	return average_accuracies, average_loss