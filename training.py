import copy
import utils
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config/yaml_file.yaml')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weight-path', type=str, default='weights/cifar10.pth')
    args = parser.parse_args()

    config = utils.load_yaml(args.config_path)

    train dataloader
    train_dataset = utils.create_instance(config['train_dataset_cifar'])
    train_loader = utils.create_instance(config['train_loader_cifar'], **{'dataset': train_dataset})

    # valid dataloader
    valid_dataset = utils.create_instance(config['valid_dataset_cifar'])
    valid_loader = utils.create_instance(config['valid_loader_cifar'], **{'dataset': valid_dataset})

    # model
    block = utils.create_instance(config['block'])
    model = utils.create_instance(config['model'], **{'block': block})

    # loss function
    criterion = utils.create_instance(config['criterion'])

    optimizer = utils.create_instance(config['optimizer'], **{'params': model.parameters()})
    lr_scheduler = utils.create_instance(config['lr_scheduler'], **{'optimizer': optimizer})

    train_epoch = utils.create_instance(config['trainer'])
    eval_epoch = utils.create_instance(config['validation'])

    best_valid_acc = 0.
    best_model_state_dict = dict()

    early_stopping = utils.create_instance(config['early_stopping'])

    if Path(args.weight_path.exists()):
        model.load_state_dict(torch.load(args.weight_path))

    for epoch in range(args.num_epochs):
        train_acc, train_loss = train_epoch(train_loader, model, criterion, optimizer, args.device)
        valid_acc, valid_loss = eval_epoch(valid_loader, model, criterion, args.device)

        print(f'#Epoch {epoch + 1}:\n- train_loss: {train_loss} - train_accuracy: {train_acc}')
        print(f'- valid_loss: {valid_loss} - valid_accuracy: {valid_acc}')

        lr_scheduler.step(valid_loss)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_acc = valid_acc

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f'Best Validation Accuracy: {best_valid_acc:4f}')

    output_dir = Path('weights')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    weight_path = output_dir.joinpath(f'best_model_accuracy={best_valid_acc}.pth')
    torch.save(obj=best_model_state_dict, f=weight_path)