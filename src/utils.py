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


def get_promptinit(config):
    if config.init_type == 'sampled':
        return PromptTuningInit.SAMPLED
    elif config.init_type == 'text':
        return PromptTuningInit.TEXT
    elif config.init_type == 'random':
        return PromptTuningInit.RANDOM
    elif config.init_type == 'class':
        return PromptTuningInit.CLASS