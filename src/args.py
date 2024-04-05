import argparse


def get_args():
    parser = argparse.ArgumentParser(
        'SPoT Replication - Prompt & Adapter evaluation')

    parser.add_argument(
        '--config-path',
        type=str,
        default='./configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='t5-multirc-finetune',
        help='Wandb project name'
    )
    parser.add_argument(
        '--wandb-model',
        type=str,
        default='checkpoint',
        help='Wandb log model'
    )
    parser.add_argument(
        '--language-adapter',
        choices=['adapter', 'prompt'],
        default='adapter'
    )
    parser.add_argument(
        '--task-adapter',
        choices=['adapter', 'prompt'],
        default='adapter'
    )
    parser.add_argument(
        '--training',
        choices=['language', 'task'],
        default='language'
    )
    parser.add_argument(
        "--use-hf",
        action='store_true',
        help='Use huggingface trainer'
    )

    return parser.parse_args()
