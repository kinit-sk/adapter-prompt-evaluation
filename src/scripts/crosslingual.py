import argparse
import os

from tasks.utils import convert_language 

default_params = [
    '--model_name_or_path bigscience/mt0-base',
    '--do_eval',
    '--predict_with_generate',
    '--per_device_eval_batch_size 32',
    '--max_seq_length 256',
    '--overwrite_output_dir',
    '--pad_to_max_length',
]

adapter_params = [
    '--train_adapter',
    '--language_adapter_type none',
    '--task_adapter_type adapter',
]

prompt_params = [
    '--language_adapter_type none',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion none'
]

adapter_adapter_params = [
    '--train_adapter',
    '--language_adapter_type adapter',
    '--task_adapter_type adapter',
]

adapter_prompt_params = [
    '--language_adapter_type adapter',
    '--task_adapter_type prompt',
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion none'
]

prompt_adapter_params = [
    '--train_adapter',
    '--language_adapter_type prompt',
    '--task_adapter_type adapter',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50'
]

prompt_prompt_params = [
    '--language_adapter_type prompt',
    '--task_adapter_type prompt',
    '--load_language_prompt ../results/language/english_prompt/english_prompt',  # Need to change
    '--prompt_tuning',
    '--task_type SEQ_2_SEQ_LM',
    '--num_virtual_tokens 50',
    '--fusion cat',
]

inference_params = [
    '--language_adapter_type none',
    '--task_adapter_type none',
    '--full_finetuning'
]


def get_dataset_from_language(language):
    if language == 'czech':
        return 'cssquad'
    elif language == 'slovak':
        return 'sksquad'
    elif language == 'english':
        return 'squad'
    else:
        return 'mlqa'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_language', type=str, default='english')
    parser.add_argument('--test_languages', nargs='+', type=str, default=['czech'])
    
    args = parser.parse_args()
    
    trained_language = args.trained_language
    test_languages = args.test_languages
    
    for language in test_languages:
        trained_dataset_name = get_dataset_from_language(trained_language)
        dataset_name = get_dataset_from_language(language)
        lang_params = [
            f'--dataset_name ',
            f'--language {language}',
        ]
        
        param_list = []
        # inference
        params = default_params + lang_params + inference_params + [f'--output_dir ../results/task/inference/{trained_language}/{language}_{dataset_name}_inference']
        param_list.append(params)
        
        # only task adapter
        params = default_params + lang_params + adapter_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_adapter_inference',
            f'--adapter_config ivykopal/{trained_language}_adapter',
            f'--load_adapter ivykopal/{trained_language}_adapter',
        ]
        param_list.append(params)
        
        # only task prompt
        params = default_params + lang_params + prompt_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_prompt_inference',
            f'--load_task_prompt ivykopal/{trained_language}_prompt',
        ]
        param_list.append(params)
        
        # adapter adapter
        params = default_params + lang_params + adapter_adapter_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_adapter_{trained_dataset_name}_adapter_inference',
            f'--lang_adapter_config ivykopal/{language}_adapter',
            f'--load_lang_adapter ivykopal/{language}_adapter',
            f'--adapter_config ivykopal/{trained_language}_adapter_{trained_dataset_name}_adapter',
            f'--load_adapter ivykopal/{trained_language}_adapter_{trained_dataset_name}_adapter',
        ]
        param_list.append(params)
        
        # adapter prompt
        params = default_params + lang_params + adapter_prompt_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_adapter_{trained_dataset_name}_prompt_inference',
            f'--lang_adapter_config ivykopal/{language}_adapter',
            f'--load_lang_adapter ivykopal/{language}_adapter',
            f'--load_task_prompt ivykopal/{trained_language}_adapter_{trained_dataset_name}_prompt',
        ]
        param_list.append(params)
        
        # prompt adapter
        params = default_params + lang_params + prompt_adapter_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_prompt_{trained_dataset_name}_adapter_inference',
            f'--load_language_prompt ivykopal/{language}_prompt',
            f'--adapter_config ivykopal/{trained_language}_prompt_{trained_dataset_name}_adapter',
            f'--load_adapter ivykopal/{trained_language}_prompt_{trained_dataset_name}_adapter',
        ]
        param_list.append(params)
        
        # prompt prompt
        params = default_params + lang_params + prompt_prompt_params + [
            f'--output_dir ../results/task/inference/{trained_language}/{language}_prompt_{dataset_name}_prompt_inference',
            f'--load_language_prompt ivykopal/{language}_prompt',
            f'--load_task_prompt ivykopal/{trained_language}_prompt_{dataset_name}_prompt',
        ]
        param_list.append(params)
        
        for params in param_list:
            os.system(
                f'python -m task_modeling.run {" ".join(params)}'
            )        
        
