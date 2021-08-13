import utils
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--config-path', type=str, default='config/cifar.yaml')
	args = parser.parse_args()

	config = utils.load_yaml(args.config_path)

	train_dataset = utils.create_instance(config['train_dataset'])
	train_loader = utils.create_instance(config['train_loader'], **{'dataset': train_dataset})

	valid_dataset = utils.create_instance(config['valid_dataset'])
	valid_loader = utils.create_instance(config['valid_loader'], **{'dataset': valid_dataset})

	visualize = utils.create_instance(config['visualize'])

	visualize(train_loader, args.device)