import argparse
import os
from tasks.utils import convert_language 

os.environ['WANDB_MODE'] = 'disabled'

default_params = [
    '--model_name_or_path bigscience/mt0-base',
    # '--do_eval',
    '--do_predict',
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


def get_dataset_from_language(language, task):
    if task == 'qa':
        if language == 'czech':
            return 'cssquad'
        elif language == 'slovak':
            return 'sksquad'
        elif language == 'telugu':
            return 'tequad'
        elif language == 'english':
            return 'mlqa'
        else:
            return 'mlqa'
    elif task == 'ner':
        return 'wikiann'
    elif task == 'nli':
        return 'xnli'
    elif task == 'checkworthy':
        return 'multiclaim_checkworthy'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_language', type=str, default='english')
    parser.add_argument('--test_languages', nargs='+', type=str, default=['czech'])
    parser.add_argument('--task', type=str, default='qa')
    
    args = parser.parse_args()
    
    trained_language = args.trained_language
    test_languages = args.test_languages
    trained_lang_code = convert_language(trained_language)
    
    for language in test_languages:
        trained_dataset_name = get_dataset_from_language(trained_language, args.task)
        dataset_name = get_dataset_from_language(language, args.task)
        lang_params = [
            f'--dataset_name {dataset_name}',
            f'--language {language}',
        ]
        
        param_list = []
        # inference
        params = default_params + lang_params + inference_params + [f'--output_dir ../results/ner2/inference2/{language}_{dataset_name}_inference']
        param_list.append(params)
        
        # only task adapter
        params = default_params + lang_params + adapter_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_adapter_inference_100k',
            f'--adapter_config ../results/ner/{trained_dataset_name}_{trained_lang_code}_adapter_100k/{trained_dataset_name}/adapter_config.json',
            f'--load_adapter ../results/ner/{trained_dataset_name}_{trained_lang_code}_adapter_100k/{trained_dataset_name}',
        ]
        param_list.append(params)
        
        # only task prompt
        params = default_params + lang_params + prompt_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_prompt_inference_100k',
            f'--load_task_prompt ../results/ner/{trained_dataset_name}_{trained_lang_code}_prompt_100k/{trained_dataset_name}_prompt',
        ]
        param_list.append(params)
        
        # adapter adapter
        params = default_params + lang_params + adapter_adapter_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_adapter_{trained_dataset_name}_adapter_inference_100k',
            f'--lang_adapter_config ivykopal/{language}_adapter_100k',
            f'--load_lang_adapter ivykopal/{language}_adapter_100k',
            f'--adapter_config ../results/ner/{trained_language}_adapter_{trained_dataset_name}_adapter_100k/{trained_dataset_name}/adapter_config.json',
            f'--load_adapter ../results/ner/{trained_language}_adapter_{trained_dataset_name}_adapter_100k/{trained_dataset_name}',
        ]
        param_list.append(params)
        
        # adapter prompt
        params = default_params + lang_params + adapter_prompt_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_adapter_{trained_dataset_name}_prompt_inference_100k',
            f'--lang_adapter_config ivykopal/{language}_adapter_100k',
            f'--load_lang_adapter ivykopal/{language}_adapter_100k',
            f'--load_task_prompt ../results/ner/{trained_language}_adapter_{trained_dataset_name}_prompt_100k/{trained_dataset_name}_prompt',
        ]
        param_list.append(params)
        
        # prompt adapter
        params = default_params + lang_params + prompt_adapter_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_prompt_{trained_dataset_name}_adapter_inference_100k',
            f'--load_language_prompt ivykopal/{language}_prompt_100k',
            f'--adapter_config ../results/ner/{trained_language}_prompt_{trained_dataset_name}_adapter_100k/adapter_config.json',
            f'--load_adapter ../results/ner/{trained_language}_prompt_{trained_dataset_name}_adapter_100k/{trained_dataset_name}',
        ]
        param_list.append(params)
        
        # prompt prompt
        params = default_params + lang_params + prompt_prompt_params + [
            f'--output_dir ../results/ner/inference/{trained_language}/{language}_prompt_{trained_dataset_name}_prompt_inference_100k',
            f'--load_language_prompt ivykopal/{language}_prompt_100k',
            f'--load_task_prompt ivykopal/{trained_language}_prompt_{trained_dataset_name}_prompt_100k',
            f'--load_task_prompt ../results/ner/{trained_language}_prompt_{trained_dataset_name}_prompt_100k/{trained_dataset_name}_prompt',
        ]
        param_list.append(params)
        
        for params in param_list:
            os.system(
                f'python -m task_modeling.run {" ".join(params)}'
            ) 
