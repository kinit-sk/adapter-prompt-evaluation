from adapters import setup_adapter_training, AdapterConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModelForCausalLM

from prompt_tuning.config import PromptTuningConfig, TaskType
from prompt_tuning.mapping import get_prompt_tuning_model
from prompt_tuning.prompt_tuning import PromptTuningForSeq2SeqLM
from utils import freeze_parameters, get_promptinit, unfreeze_parameters


def get_model(model_args, config, task=None):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return model


def get_updated_model(model, model_args, adapter_args, prompt_args, dataset_name=None):
    language_adapter_type = model_args.language_adapter_type
    task_adapter_type = model_args.task_adapter_type

    if language_adapter_type == 'none' and task_adapter_type == 'none':
        return model
    elif language_adapter_type == 'none' and task_adapter_type == 'adapter':
        setup_adapter_training(model, adapter_args, dataset_name or "mlm")
        return model
    elif language_adapter_type == 'adapter' and task_adapter_type == 'adapter':
        setup_adapter_training(model, adapter_args, dataset_name or "mlm")
        return model
    elif language_adapter_type == 'none' and task_adapter_type == 'prompt':
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=get_promptinit(prompt_args.prompt_tuning_init),
            num_virtual_tokens=prompt_args.num_virtual_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
            prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
            fusion=prompt_args.fusion
        )

        model = get_prompt_tuning_model(
            model, peft_config=peft_config, adapter_name=f'{dataset_name}_prompt')

        model = freeze_parameters(model)
        unfreeze_parameters(model, f'{dataset_name}_prompt')
        print(model)
        return model
    elif language_adapter_type == 'adapter' and task_adapter_type == 'prompt':
        lang_adapter_config = AdapterConfig.load(
            adapter_args.lang_adapter_config)
        # load the language adapter from Hub
        lang_adapter_name = model.load_adapter(
            adapter_args.load_lang_adapter,
            config=lang_adapter_config,
        )
        model.set_active_adapters(lang_adapter_name)

        if model_args.load_task_prompt is not None:
            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, model_args.load_task_prompt, adapter_name=f'{prompt_args.language}_prompt')
        else:
            peft_task_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(
                    prompt_args.prompt_tuning_init),
                num_virtual_tokens=prompt_args.num_virtual_tokens,
                tokenizer_name_or_path=model_args.model_name_or_path,
                prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
                task_prompt=True,
                fusion=prompt_args.fusion
            )

            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{dataset_name}_prompt')
            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{dataset_name}_prompt')

        print(model)
        return model

    elif language_adapter_type == 'prompt' and task_adapter_type == 'adapter':

        adapter_config = AdapterConfig.load(adapter_args.adapter_config)

        if dataset_name not in model.adapters_config:
            model.add_adapter(dataset_name, config=adapter_config)

        model = PromptTuningForSeq2SeqLM.from_pretrained(
            model, model_args.load_language_prompt, adapter_name=f'{prompt_args.language}_prompt')
        model = freeze_parameters(model)
        model.train_adapter([dataset_name])
        model.set_active_adapters(dataset_name)

        print(model.active_adapters)
        print(model)
        return model
    elif language_adapter_type == 'prompt' and task_adapter_type == 'prompt' and prompt_args.prompt_tuning:
        peft_task_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=get_promptinit(prompt_args.prompt_tuning_init),
            num_virtual_tokens=prompt_args.num_virtual_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
            prompt_tuning_init_text=prompt_args.prompt_tuning_init_text,
            task_prompt=True,
            fusion=prompt_args.fusion
        )

        model = PromptTuningForSeq2SeqLM.from_pretrained(
            model, model_args.load_language_prompt, adapter_name=f'{prompt_args.language}_prompt')

        model = get_prompt_tuning_model(
            model, peft_config=peft_task_config, adapter_name=f'{dataset_name}_prompt')

        model = freeze_parameters(model)
        unfreeze_parameters(model, f'{dataset_name}_prompt')
        print(model)
        return model
    else:
        raise ValueError("Invalid adapter configuration")
