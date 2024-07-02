import configparser as ConfigParser
import os

default_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--do_train',
    '--do_eval',
    '--do_predict',
    '--evaluation_strategy steps',
    '--eval_steps 1000',
    '--save_steps 1000',
    '--max_answer_length 4',
    '--load_best_model_at_end',
    '--predict_with_generate',
    '--per_device_train_batch_size 32',
    '--per_device_eval_batch_size 32',
    '--max_steps 50000',
    '--max_seq_length 256',
    '--overwrite_output_dir',
    '--pad_to_max_length',
    '--seed 42',
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
    '--num_virtual_tokens 50',
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
    '--num_virtual_tokens 50',
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
    '--num_virtual_tokens 50'
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

inference_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--do_predict',
    '--predict_with_generate',
    '--per_device_eval_batch_size 32',
    '--max_seq_length 256',
    '--max_answer_length 4',
    '--overwrite_output_dir',
    '--pad_to_max_length',
    '--language_adapter_type none',
    '--task_adapter_type none',
    '--full_finetuning'
]


if __name__ == '__main__':
    languages = ['english', 'slovak', 'czech', 'german', 'spanish', 'telugu']
    lang_code = ['en', 'sk', 'cs', 'de', 'es', 'te']
    datasets = ['multiclaim_checkworthy'] * 6

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
        os.environ['WANDB_NAME'] = f'mt0-{dataset}-{code}-adapter-100k'
        params = default_params + adapter_params + lang_params + \
            [f'--output_dir ../results/checkworthy/{dataset}_{code}_adapter_100k']
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )

        # train only task prompt
        os.environ['WANDB_NAME'] = f'mt0-{dataset}-{code}-prompt-100k'
        params = default_params + prompt_params + lang_params + [
            f'--output_dir ../results/checkworthy/{dataset}_{code}_prompt_100k',
            f'--prompt_tuning_init_text "Determine whether a given claim in {language.capitalize()} is checkworthy:"',]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )

        # train task adapter with language adapter
        os.environ['WANDB_NAME'] = f'mt0-{language}-adapter-{dataset}-adapter-100k'
        params = default_params + adapter_adapter_params + lang_params + [
            f'--output_dir ../results/checkworthy/{language}_adapter_{dataset}_adapter_100k',
            f'--lang_adapter_config ivykopal/{language}_adapter_100k',
            f'--load_lang_adapter ivykopal/{language}_adapter_100k',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )

        # train language adapter with task prompt
        os.environ['WANDB_NAME'] = f'mt0-{language}-adapter-{dataset}-prompt-100k'
        params = default_params + adapter_prompt_params + lang_params + [
            f'--output_dir ../results/checkworthy/{language}_adapter_{dataset}_prompt_100k',
            f'--lang_adapter_config ivykopal/{language}_adapter_100k',
            f'--load_lang_adapter ivykopal/{language}_adapter_100k',
            f'--prompt_tuning_init_text "Determine whether a given claim in {language.capitalize()} is checkworthy:"',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )

        # train language prompt with task adapter
        os.environ['WANDB_NAME'] = f'mt0-{language}-prompt-{dataset}-adapter-100k'
        params = default_params + prompt_adapter_params + lang_params + [
            f'--output_dir ../results/checkworthy/{language}_prompt_{dataset}_adapter_100k',
            f'--load_language_prompt ivykopal/{language}_prompt_100k',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )

        # train task prompt with language prompt
        os.environ['WANDB_NAME'] = f'mt0-{language}-prompt-{dataset}-prompt-100k'
        params = default_params + prompt_prompt_params + lang_params + [
            f'--output_dir ../results/checkworthy/{language}_prompt_{dataset}_prompt_100k',
            f'--load_language_prompt ivykopal/{language}_prompt_100k',
            f'--prompt_tuning_init_text "Determine whether a given claim in {language.capitalize()} is checkworthy:"',
        ]
        os.system(
            f'python -m task_modeling.run {" ".join(params)}'
        )
