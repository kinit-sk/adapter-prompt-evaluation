import os
import argparse

default_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--dataset_name sksquad'
    '--do_train',
    '--do_eval',
    '--predict_with_generate',
    '--per_device_train_batch_size 32',
    '--per_device_eval_batch_size 32',
    '--max_steps 50000',
    '--max_seq_length 256',
    '--overwrite_output_dir',
    '--pad_to_max_length',
    '--report_to wandb',
    '--language slovak',
]

adapter_params = [
    '--learning_rate 5e-5',
    '--output_dir ../results/task/sksquad_adapter',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--language_adapter_type none',
    '--task_adapter_type adapter',
]

prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--output_dir ../results/task/sk_squad_prompt',
    '--optim adafactor',
    '--language_adapter_type none',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion none'
]

adapter_adapter_params = [
    '--learning_rate 5e-5',
    '--output_dir ../results/task/english_adapter_squad_adapter',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--language_adapter_type adapter',
    '--task_adapter_type adapter',
    '--lang_adapter_config ../results/language/wikipedia/adapter_config.json',  # Need to change
    '--load_lang_adapter ../results/language/wikipedia',  # Need to change
]

adapter_prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--output_dir ../results/task/english_adapter_squad_prompt',
    '--language_adapter_type adapter',
    '--optim adafactor',
    '--task_adapter_type prompt',
    '--lang_adapter_config ../results/language/english_adapter/wikipedia/adapter_config.json',  # Need to change
    '--load_lang_adapter ../results/language/english_adapter/wikipedia',  # Need to change
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion none'
]

prompt_adapter_params = [
    '--learning_rate 5e-5',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--output_dir ../results/task/english_prompt_squad_adapter',  # Need to change
    '--language_adapter_type prompt',
    '--task_adapter_type adapter',
    '--load_language_prompt ../results/language/english_prompt/english_prompt',  # Need to change
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50'
]

prompt_prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--optim adafactor',
    '--output_dir ../results/task/english_prompt_squad_prompt',  # Need to change
    '--language_adapter_type prompt',
    '--task_adapter_type prompt',
    '--load_language_prompt ../results/language/english_prompt/english_prompt',  # Need to change
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion cat',
]


os.environ['WANDB_PROJECT'] = 'squad'
os.environ['WANDB_WATCH'] = 'all'
# os.environ['WANDB_MODE'] = 'disabled'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Task Training Script')
    parser.add_argument('--language_type', type=str, default='adapter',
                        help='Type of script to run')
    parser.add_argument('--task_type', type=str, default='adapter',
                        help='Type of script to run')
    args = parser.parse_args()

    if args.language_type == 'none' and args.task_type == 'adapter':
        hyper_params = default_params + adapter_params
        os.environ['WANDB_NAME'] = 'mt0-sksquad_adapter'
    elif args.language_type == 'none' and args.task_type == 'prompt':
        hyper_params = default_params + prompt_params
        os.environ['WANDB_NAME'] = 'mt0-sksquad_prompt'
    elif args.language_type == 'adapter' and args.task_type == 'adapter':
        hyper_params = default_params + adapter_adapter_params
        os.environ['WANDB_NAME'] = 'mt0-slovak-adapter_squad_adatper'
    elif args.language_type == 'adapter' and args.task_type == 'prompt':
        hyper_params = default_params + adapter_prompt_params
        os.environ['WANDB_NAME'] = 'mt0-slovak-adapter_squad_prompt'
    elif args.language_type == 'prompt' and args.task_type == 'adapter':
        hyper_params = default_params + prompt_adapter_params
        os.environ['WANDB_NAME'] = 'mt0-slovak-prompt_squad_adapter'
    elif args.language_type == 'prompt' and args.task_type == 'prompt':
        hyper_params = default_params + prompt_prompt_params
        os.environ['WANDB_NAME'] = 'mt0-slovak-prompt_squad_prompt'

    os.system(
        f'python -m task_modeling.run {" ".join(hyper_params)}'
    )
