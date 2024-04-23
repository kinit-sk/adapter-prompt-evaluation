import configparser as ConfigParser
import os

default_params = [
    '--model_name_or_path bigscience/mt0-base',
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
]

adapter_params = [
    '--learning_rate 5e-5',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--language_adapter_type none',
    '--task_adapter_type adapter',
]

prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--optim adafactor',
    '--language_adapter_type none',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 100',
    '--prompt_tuning_init text',
    '--fusion none'
]

adapter_adapter_params = [
    '--learning_rate 5e-5',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--language_adapter_type adapter',
    '--task_adapter_type adapter',
]

adapter_prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--language_adapter_type adapter',
    '--optim adafactor',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 100',
    '--prompt_tuning_init text',
    '--fusion none'
]

prompt_adapter_params = [
    '--learning_rate 5e-5',
    '--train_adapter',
    '--adapter_config seq_bn',
    '--language_adapter_type prompt',
    '--task_adapter_type adapter',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 100'
]

prompt_prompt_params = [
    '--learning_rate 5e-1',
    '--weight_decay 0.00001',
    '--optim adafactor',
    '--language_adapter_type prompt',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--prompt_tuning_init text',
    '--fusion cat',
]


def publish_to_hf(name, folder_path):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(
        repo_id=f'ivykopal/{name}',
        token=os.getenv('HF_TOKEN'),
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=folder_path,
        repo_id=f'ivykopal/{name}',
        repo_type="model",
        token=os.getenv('HF_TOKEN'),
    )


if __name__ == '__main__':
    # Script only for the training task adapters/prompts
    languages = ['english', 'slovak', 'czech', 'german', 'spanish']
    lang_code = ['en', 'sk', 'cs', 'de', 'es']
    datasets = ['mlqa', 'sksquad', 'cssquad', 'mlqa', 'mlqa']

    os.environ['WANDB_WATCH'] = 'all'

    config = ConfigParser.ConfigParser()
    config.read('../configs/api.conf')
    os.environ['HF_TOKEN'] = config.get('huggingface', 'HF_API_KEY')

    for dataset, code, language in zip(datasets, lang_code, languages):
        os.environ['WANDB_PROJECT'] = dataset
        lang_params = [
            f'--dataset_name {dataset}',
            f'--language {language}',
        ]

        # train only task adapter
        os.environ['WANDB_NAME'] = f'mt0-{dataset}-{code}-adapter'
        params = default_params + adapter_params + lang_params + \
            [f'--output_dir ../results/task/{dataset}_{code}_adapter']
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{dataset}_{code}_adapter',
                      f'../results/task/{dataset}_{code}_adapter/{dataset}')

        # train only task prompt
        os.environ['WANDB_NAME'] = f'mt0-{dataset}-{code}-prompt-100'
        params = default_params + prompt_params + lang_params + \
            [f'--output_dir ../results/task/{dataset}_{code}_prompt_100',
                f'--prompt_tuning_init_text "Answer the question in {language.capitalize()}:"',]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{dataset}_{code}_prompt_100',
                      f'../results/task/{dataset}_{code}_prompt_100/{dataset}_prompt')

        # train task adapter with language adapter
        os.environ['WANDB_NAME'] = f'mt0-{language}-adapter-{dataset}-adapter'
        params = default_params + adapter_adapter_params + lang_params + [
            f'--output_dir ../results/task/{language}_adapter_{dataset}_adapter',
            f'--lang_adapter_config ../results/language/{language}_adapter',
            f'--load_lang_adapter ../results/language/{language}_adapter',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_adapter_{dataset}_adapter',
                      f'../results/task/{language}_adapter_{dataset}_adapter/{dataset}')

        # train task adapter with language prompt
        os.environ['WANDB_NAME'] = f'mt0-{language}-adapter-{dataset}-prompt-100'
        params = default_params + adapter_prompt_params + lang_params + [
            f'--output_dir ../results/task/{language}_adapter_{dataset}_prompt_100',
            f'--lang_adapter_config ../results/language/{language}_adapter',
            f'--load_lang_adapter ../results/language/{language}_adapter',
            f'--prompt_tuning_init_text "Answer the question in {language.capitalize()}:"',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_adapter_{dataset}_prompt_100',
                      f'../results/task/{language}_adapter_{dataset}_prompt_100/{dataset}_prompt')

        # train task prompt with language adapter
        os.environ['WANDB_NAME'] = f'mt0-{language}-prompt-{dataset}-adapter-100'
        params = default_params + prompt_adapter_params + lang_params + [
            f'--output_dir ../results/task/{language}_prompt_{dataset}_adapter_100',
            f'--load_language_prompt ../results/language/{language}_prompt_100',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_prompt_{dataset}_adapter_100',
                      f'../results/task/{language}_prompt_{dataset}_adapter_100/{dataset}')

        # train task prompt with language prompt
        os.environ['WANDB_NAME'] = f'mt0-{language}-prompt-{dataset}-prompt-100'
        params = default_params + prompt_prompt_params + lang_params + [
            f'--output_dir ../results/task/{language}_prompt_{dataset}_prompt_100',
            f'--load_language_prompt ../results/language/{language}_prompt',
            f'--prompt_tuning_init_text "Answer the question in {language.capitalize()}:"',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
        publish_to_hf(f'{language}_prompt_{dataset}_prompt_100',
                      f'../results/task/{language}_prompt_{dataset}_prompt-100/{dataset}_prompt')
