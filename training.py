import copy
import utils
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='config/hymenoptera_training.yaml')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = utils.load_yaml(args.config_path)

    # train dataloader
    train_dataset = utils.create_instance(config['train_dataset'])
    train_loader = utils.create_instance(config['train_loader'], **{'dataset': train_dataset})

    # train eval dataloader
    train_eval_dataset = utils.create_instance(config['train_eval_dataset'])
    train_eval_loader = utils.create_instance(config['train_eval_loader'], **{'dataset': train_eval_dataset})

    # valid dataloader
    valid_dataset = utils.create_instance(config['valid_dataset'])
    valid_loader = utils.create_instance(config['valid_loader'], **{'dataset': valid_dataset})

    # model
    model = utils.create_instance(config['model'])

    # loss function
    criterion = utils.create_instance(config['criterion'])

    optimizer = utils.create_instance(config['optimizer'], **{'params': model.parameters()})
    lr_scheduler = utils.create_instance(config['lr_scheduler'], **{'optimizer': optimizer})

    train_epoch = utils.create_instance(config['trainer'])
    eval_epoch = utils.create_instance(config['evaluator'])

    train_loss_history = []
    train_acc_history = []

    train_eval_loss_history = []
    train_eval_acc_history = []

    valid_loss_history = []
    valid_acc_history = []

    best_valid_acc = 0.
    best_model_state_dict = dict()
    best_optim_state_dict = dict()


    for epoch in range(args.num_epochs):
        train_acc, train_loss = train_epoch(train_loader, model, criterion, optimizer, args.device)
        valid_acc, valid_loss = eval_epoch(valid_loader, model, criterion, args.device)

        print(f'#Epoch {epoch + 1}:\n- train_loss: {train_loss} - train_accuracy: {train_acc}')
        print(f'- valid_loss: {valid_loss} - valid_accuracy: {valid_acc}')

        lr_scheduler.step(valid_loss)

        if valid_acc > best_valid_acc:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_optim_state_dict = copy.deepcopy(optimizer.state_dict())
            best_valid_acc = valid_acc


        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)
        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)

    print(f'Best Validation Accuracy: {best_valid_acc:4f}')

    output_dir = Path('weights')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    weight_path = output_dir.joinpath(f'best_model_accuracy={best_valid_acc}.pth')
    torch.save(obj=best_model_state_dict, f=weight_path)