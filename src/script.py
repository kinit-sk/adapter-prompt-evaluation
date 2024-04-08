import os
import argparse

adapter_hyper_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--dataset_name wikipedia',
    '--dataset_config_name 20231101.en',
    '--per_device_train_batch_size 32',
    '--per_device_eval_batch_size 32',
    '--optim adafactor',
    '--learning_rate 5e-5',
    '--weight_decay 0',
    '--do_train',
    '--do_eval',
    '--output_dir ../results/language',
    '--adapter_config seq_bn',
    '--train_adapter',
    '--t5_modeling',
    '--overwrite_output_dir',
    '--max_seq_length 256',
    '--pad_to_max_length',
    '--max_steps 50000',
    '--report_to wandb',
]

prompt_hyper_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--dataset_name wikipedia',
    '--dataset_config_name 20231101.en',
    '--per_device_train_batch_size 32',
    '--per_device_eval_batch_size 32',
    '--optim adafactor',
    '--learning_rate 5e-5',
    '--weight_decay 0.00001',
    '--do_train',
    '--do_eval',
    '--output_dir ../results/language/english_prompt',
    '--t5_modeling',
    '--overwrite_output_dir',
    '--max_seq_length 256',
    '--pad_to_max_length',
    '--max_steps 50000',
    '--report_to wandb',
    '--language english',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50'
]


os.environ['WANDB_PROJECT'] = 'mt0-language-modeling'
os.environ['WANDB_WATCH'] = 'all'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Language Modeling Training Script')
    parser.add_argument('--type', type=str, default='adapter',
                        help='Type of script to run')
    args = parser.parse_args()

    if args.type == 'adapter':
        os.environ['WANDB_NAME'] = 'mt0-english-adapter'
        os.system(
            f'python -m language_modeling.run {" ".join(adapter_hyper_params)}'
        )
    else:
        os.environ['WANDB_NAME'] = 'mt0-english-prompt'
        os.system(
            f'python -m language_modeling.run {" ".join(prompt_hyper_params)}'
        )
