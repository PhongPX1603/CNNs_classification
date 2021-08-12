import argparse
import utils

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-path', type=str, default='config/config_yaml.yaml')
	parser.add_argument('--num-epochs', type=int, default=50)
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	config = utils.load_yaml(args.config_path)

	train_data = utils.create_instance(config['train_dataset'])
	train_loader = utils.create_instance(config['train_loader'], **{'dataset': train_data})

	valid_data = utils.create_instance(config['valid_dataset'])
	valid_loader = utils.create_instance(config['valid_loader'], **{'dataset': valid_data})

	model = utils.create_instance(config['model'])

	optimizer = utils.create_instance(config['optimizer'], **{'params': model.parameters()})

	criterion = utils.create_instance(config['criterion'])

	lr_scheduler = utils.create_instance(config['lr_scheduler'], **{'optimizer': optimizer})

	train_epoch = utils.create_instance(config['trainer'])
	valid_epoch = utils.create_instance(config['validation'])


	best_acc = 0
	for epoch in range(args.num_epochs):
		acc_train, loss_train = train_epoch(train_loader, model, criterion, optimizer, args.device)
		acc_valid, loss_valid = valid_epoch(valid_loader, model, criterion, args.device)
		print(f'epoch {epoch}:\n- train_loss: {loss_train} - train_acc: {acc_train}')
		print(f'- valid_loss: {loss_valid} - valid_acc: {acc_valid}')

		if acc_valid > best_acc:
			best_acc = acc_valid

	print(f'best accuracy: {best_acc}')