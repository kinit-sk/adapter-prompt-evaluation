import configparser as ConfigParser
import os
from scripts.qa import publish_to_hf

default_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--dataset_name wikipedia',
    '--per_device_train_batch_size 32',
    '--per_device_eval_batch_size 32',
    '--do_train',
    '--do_eval',
    '--t5_modeling',
    '--overwrite_output_dir',
    '--max_seq_length 256',
    '--pad_to_max_length',
    '--max_steps 50000',
    '--report_to wandb',
    '--max_eval_samples 500000',
]

adapter_hyper_params = [
    '--learning_rate 5e-5',
    '--weight_decay 0',
    '--adapter_config seq_bn',
    '--train_adapter',
]

prompt_hyper_params = [
    '--optim adafactor',
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 100',
    '--prompt_tuning_init text',
]


os.environ['WANDB_PROJECT'] = 'mt0-language-modeling'
os.environ['WANDB_WATCH'] = 'all'


if __name__ == '__main__':
    # ['english', 'slovak', 'czech', 'german', 'spanish']
    languages = ['arabic']
    lang_codes = ['ar']  # ['en', 'sk', 'cs', 'de', 'es']

    config = ConfigParser.ConfigParser()
    config.read('../configs/api.conf')
    os.environ['HF_TOKEN'] = config.get('huggingface', 'HF_API_KEY')

    for code, language in zip(lang_codes, languages):
        lang_params = [
            f'--dataset_config_name 20231101.{code}',
            f'--language {language}'
        ]
        os.environ['WANDB_NAME'] = f'mt0-{language}-adapter'
        params = default_params + adapter_hyper_params + lang_params + \
            [f'--output_dir ../results/language/{language}_adapter',]
        os.system(
            f'python -m language_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_adapter',
                      f'../results/language/{language}_adapter/wikipedia')

        os.environ['WANDB_NAME'] = f'mt0-{language}-prompt-100'
        params = default_params + prompt_hyper_params + lang_params + \
            [
                f'--output_dir ../results/language/{language}_prompt_100',
                f'--prompt_tuning_init_text "Generate the output in the {language.capitalize()} language:"',
            ]
        os.system(
            f'python -m language_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_prompt_100',
                      f'../results/language/{language}_prompt_100/{language}_prompt')
