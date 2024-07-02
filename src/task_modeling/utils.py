from adapters import setup_adapter_training, AdapterConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM
import os

from prompt_tuning.config import PromptTuningConfig, TaskType
from prompt_tuning.mapping import get_prompt_tuning_model
from prompt_tuning.prompt_tuning import PromptTuningForSeq2SeqLM
from utils import freeze_parameters, get_promptinit, unfreeze_parameters, download_model


def get_model(model_args, config, task=None):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model


def get_path(path, type='adapter'):
    if os.path.exists(path):
        return path
    else:
        return download_model(path, type)


def get_updated_model(model, model_args, adapter_args, prompt_args, dataset_name=None):
    language_adapter_type = model_args.language_adapter_type
    task_adapter_type = model_args.task_adapter_type

    if language_adapter_type == 'none' and task_adapter_type == 'none':
        return model
    elif language_adapter_type == 'none' and task_adapter_type == 'adapter':
        if adapter_args.load_adapter:
            load_adapter = get_path(adapter_args.load_adapter)
            adapter_args.adapter_config = f'{load_adapter}/adapter_config.json'
            adapter_args.load_adapter = load_adapter
            
        print(adapter_args)

        setup_adapter_training(model, adapter_args, dataset_name or "mlm")
        return model
    elif language_adapter_type == 'adapter' and task_adapter_type == 'adapter':
        load_lang_adapter = get_path(adapter_args.load_lang_adapter)
        adapter_args.lang_adapter_config = f'{load_lang_adapter}/adapter_config.json'
        adapter_args.load_lang_adapter = load_lang_adapter
        if adapter_args.load_adapter:
            load_adapter = get_path(adapter_args.load_adapter)
            adapter_args.adapter_config = f'{load_adapter}/adapter_config.json'
            adapter_args.load_adapter = load_adapter
            
        print(adapter_args)
        setup_adapter_training(model, adapter_args, dataset_name or "mlm")
        return model
    elif language_adapter_type == 'none' and task_adapter_type == 'prompt':
        if model_args.load_task_prompt is not None:
            load_task_prompt = get_path(model_args.load_task_prompt, type="prompt")
            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, load_task_prompt, adapter_name=f'{dataset_name}_prompt')
        else:
            peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(prompt_args.prompt_tuning_init),
                num_virtual_tokens=prompt_args.num_virtual_tokens,
                tokenizer_name_or_path=model_args.model_name_or_path,
                prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
                fusion=prompt_args.fusion,
                partial_embedding=prompt_args.partial_embedding,
                fixed_size=prompt_args.fixed_size,
                partial_prompt_tuning_init_text=prompt_args.partial_prompt_tuning_init_text,
            )

            model = get_prompt_tuning_model(
                model, peft_config=peft_config, adapter_name=f'{dataset_name}_prompt')

            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{dataset_name}_prompt')
        return model
    elif language_adapter_type == 'adapter' and task_adapter_type == 'prompt':
        load_lang_adapter = get_path(adapter_args.load_lang_adapter)
        lang_adapter_config = AdapterConfig.load(
            f'{load_lang_adapter}/adapter_config.json')
        # load the language adapter from Hub
        lang_adapter_name = model.load_adapter(
            load_lang_adapter,
            config=lang_adapter_config,
        )
        model.set_active_adapters(lang_adapter_name)

        if model_args.load_task_prompt is not None:
            load_task_prompt = get_path(model_args.load_task_prompt, type="prompt")
            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, load_task_prompt, adapter_name=f'{dataset_name}_prompt')
        else:
            peft_task_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(
                    prompt_args.prompt_tuning_init),
                num_virtual_tokens=prompt_args.num_virtual_tokens,
                tokenizer_name_or_path=model_args.model_name_or_path,
                prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
                task_prompt=True,
                fusion=prompt_args.fusion,
                partial_embedding=prompt_args.partial_embedding,
                fixed_size=prompt_args.fixed_size,
                partial_prompt_tuning_init_text=prompt_args.partial_prompt_tuning_init_text,
            )

            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{dataset_name}_prompt')
            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{dataset_name}_prompt')

        return model
    elif language_adapter_type == 'prompt' and task_adapter_type == 'adapter':

        load_language_prompt = get_path(model_args.load_language_prompt, type="prompt")

        if adapter_args.load_adapter:
            load_adapter = get_path(adapter_args.load_adapter)
            print(adapter_args, load_adapter)
            adapter_config = AdapterConfig.load(f'{load_adapter}/adapter_config.json')
            adapter_name = model.load_adapter(
                load_adapter,
                config=adapter_config,
            )
            model.set_active_adapters(adapter_name)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, load_language_prompt, adapter_name=f'{prompt_args.language}_prompt')
        else:

            adapter_config = AdapterConfig.load(adapter_args.adapter_config)
            if dataset_name not in model.adapters_config:
                model.add_adapter(dataset_name, config=adapter_config)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, load_language_prompt, adapter_name=f'{prompt_args.language}_prompt')
            model = freeze_parameters(model)
            model.train_adapter([dataset_name])
            model.set_active_adapters(dataset_name)

        return model
    elif language_adapter_type == 'prompt' and task_adapter_type == 'prompt':
        load_language_prompt = get_path(
            model_args.load_language_prompt, type="prompt")
        print("Loading language prompt")
        model = PromptTuningForSeq2SeqLM.from_pretrained(
            model, load_language_prompt, adapter_name=f'{prompt_args.language}_prompt')

        print("Loading task prompt")
        if model_args.load_task_prompt is not None:
            load_task_prompt = get_path(model_args.load_task_prompt, type="prompt")
            print(load_language_prompt, load_task_prompt)
            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, load_task_prompt, adapter_name=f'{dataset_name}_prompt')
            # config = PromptTuningConfig.from_pretrained(load_task_prompt)
            # model.add_adapter(f'{dataset_name}_prompt', prompt_config=config)
            # model.load_adapter(load_task_prompt, adapter_name=f'{dataset_name}_prompt')
        else:
            peft_task_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(
                    prompt_args.prompt_tuning_init),
                num_virtual_tokens=prompt_args.num_virtual_tokens,
                tokenizer_name_or_path=model_args.model_name_or_path,
                prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
                task_prompt=True,
                fusion=prompt_args.fusion,
                partial_embedding=prompt_args.partial_embedding,
                fixed_size=prompt_args.fixed_size,
                partial_prompt_tuning_init_text=prompt_args.partial_prompt_tuning_init_text,
            )
            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{dataset_name}_prompt')
            # model.add_adapter(f'{dataset_name}_prompt', prompt_config=peft_task_config)
            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{dataset_name}_prompt')
        return model
    else:
        raise ValueError("Invalid adapter configuration")
