import configparser as ConfigParser
from collections import namedtuple
import yaml


WandbConfig = namedtuple(
    'WandbConfig', ['WANDB_API_KEY', 'WANDB_USERNAME', 'WANDB_DIR'])

HuggingfaceConfig = namedtuple(
    'HuggingfaceConfig', ['HF_API_KEY'])


def get_wandb_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    return WandbConfig(
        WANDB_API_KEY=config.get('wandb', 'WANDB_API_KEY'),
        WANDB_USERNAME=config.get('wandb', 'WANDB_USERNAME'),
        WANDB_DIR=config.get('wandb', 'WANDB_DIR')
    )


def get_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config = namedtuple('config', config.keys())(
        *config.values())
    return config


def get_huggingface_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    return HuggingfaceConfig(
        HF_API_KEY=config.get('huggingface', 'HF_API_KEY'),
    )
