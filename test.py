import utils
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	config = utils.load_yaml('config/cifar.yaml')

	train_dataset = utils.create_instance(config['train_dataset'])
	train_loader = utils.create_instance(config['train_loader'])

	valid_dataset = utils.create_instance(config['valid_dataset'])
	valid_loader = utils.create_instance(config['valid_loader'])

	visualize = utils.create_instance(config['visualize'])

	visualize(train_loader, args.device)