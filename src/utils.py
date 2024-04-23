import os
from prompt_tuning.config import PromptTuningInit


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze_parameters(model, name):
    for module_name, module in model.named_children():
        if module_name == name:
            for param in module.parameters():
                param.requires_grad = True
        else:
            unfreeze_parameters(module, name)


def get_train_type(args):
    language_adapter = args.language_adapter
    task_adapter = args.task_adapter
    training = args.training

    if training == 'language' and language_adapter == 'adapter':
        return 'adapter'
    elif training == 'language' and task_adapter == 'prompt':
        return 'prompt'
    elif training == 'task' and task_adapter == 'adapter':
        return 'adapter'
    elif training == 'task' and task_adapter == 'prompt':
        return 'prompt'


def get_promptinit(init_type):
    if init_type == 'sampled':
        return PromptTuningInit.SAMPLED
    elif init_type == 'text':
        return PromptTuningInit.TEXT
    elif init_type == 'random':
        return PromptTuningInit.RANDOM
    elif init_type == 'class':
        return PromptTuningInit.CLASS


def download_model(model_name, type='adapter'):
    if os.path.exists(f'../cache/models/{model_name}'):
        return f'../cache/models/{model_name}'

    os.mkdir(f'../cache/models/{model_name}', exist_ok=True)
    if type == 'adapter':
        files = [
            'adapter_config.json',
            'head_config.json',
            'pytorch_adapter.bin',
            'pytorch_model_head.bin',
        ]
    elif type == 'prompt':
        files = [
            'adapter_config.json',
            'adapter_model.bin',
        ]

    for file in files:
        output_path = f'../cache/models/{model_name}/{file}'
        os.system(
            f'curl -Ls -o {output_path} https://huggingface.co/{model_name}/resolve/main/{file}?download=true'
        )

    return f'../cache/models/{model_name}'
