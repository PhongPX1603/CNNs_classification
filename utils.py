from importlib import import_module
import torchvision
import yaml


def load_yaml(yaml_file):
	with open(yaml_file, mode='r', encoding='utf-8') as f:
		configs = yaml.safe_load(f)
	return configs


def create_instance(config, *args, **kwargs):
	module = config['module']
	name = config['class']
	config_kwargs = config.get(name, {})
	for key, value in config_kwargs.item():
		if isinstance(value, str):
			config_kwargs[key] = eval(value)
		if isinstance(value, list):
			config_kwargs[key] = [eval(v) for v in value]
		if isinstance(value, tuple):
			config_kwargs[key] = tuple([eval(v) for v in value])

	return getatrr(import_module(module), name)(*arg, **config_kwargs, **kwargs)