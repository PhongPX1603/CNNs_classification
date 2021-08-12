import torch
from torch import nn


class TrainEpoch(nn.Module):
    def __init__(self):
        super(TrainEpoch, self).__init__()

    def forward(self, data_loader, model, criterion, optimizer, device):
        model.to(device)
        model.train()  # set training mode

        accuracies, losses = [], []
        for x, y_true in data_loader:
            x = x.to(device)
            y_true = y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()

            y_pred = nn.Softmax(dim=1)(y_pred)
            y_pred = y_pred.argmax(dim=1)  # B, num_classes
            correct = torch.sum(y_true == y_pred)
            accuracy = correct.item() / y_true.shape[0]

            accuracies.append(accuracy)
            losses.append(loss.item())

        average_loss = sum(losses) / len(losses) if len(losses) else 0.
        average_accuracy = sum(accuracies) / len(accuracies) if len(accuracies) else 0.

        return average_accuracy, average_loss


class EvalEpoch(nn.Module):
    def __init__(self):
        super(EvalEpoch, self).__init__()

    def forward(self, data_loader, model, criterion, device):
        model.to(device)
        model.eval()    # set valuating mode

        accuracies, losses = [], []

        with torch.no_grad():
            for x, y_true in data_loader:
                x = x.to(device) 
                y_true = y_true.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y_true)

                y_pred = nn.Softmax(dim=1)(y_pred)
                y_pred = y_pred.argmax(dim=1)  # B, num_classes

                correct = torch.sum(y_true == y_pred)
                accuracy = correct.item() / y_true.shape[0]

                accuracies.append(accuracy)
                losses.append(loss.item())

        average_loss = sum(losses) / len(losses) if len(losses) else 0.
        average_accuracy = sum(accuracies) / len(accuracies) if len(accuracies) else 0.

        return average_accuracy, average_loss