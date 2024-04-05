from args import get_args
from adapters import AutoAdapterModel, AdapterConfig
import logging
import os
from transformers import AutoConfig, AutoTokenizer, default_data_collator, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from typing import Dict, Any

from prompt_tuning.config import PromptTuningConfig, TaskType
from prompt_tuning import get_prompt_tuning_model
from prompt_tuning.prompt_tuning import PromptTuningForSeq2SeqLM
from tasks.tasks import TaskDataset
from train.sampling import all_mixing, proportional_mixing
from train.trainer import CustomTrainer
from config_utils import get_config, get_huggingface_config, get_wandb_config
from config import Config
from utils import freeze_parameters, get_promptinit, get_train_type, unfreeze_parameters


logging.basicConfig(level=logging.INFO)


def get_model_tokenizer(args: Dict, model_name_or_path: str, config: Any, task_name: str):
    language_adapter_type = args.language_adapter
    task_adapter_type = args.task_adapter
    training_type = args.training
    language = config.language

    config = AutoConfig.from_pretrained(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoAdapterModel.from_pretrained(model_name_or_path, config=config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_type == 'language':
        logging.info(
            f"Training language adapter for {language}")
        if language_adapter_type == 'adapter':
            adapter_config = AdapterConfig.load(
                "seq_bn_inv", reduction_factor=2)
            model.add_adapter(f'{language}_adapter', config=adapter_config)
            model.train_adapter(f'{language}_adapter')
            model.set_active_adapters(f'{language}_adapter')
            logging.info(f"Active adapters: {model.active_adapters}")
        elif language_adapter_type == 'prompt':
            peft_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(config),
                num_virtual_tokens=config.num_virtual_tokens,
                tokenizer_name_or_path=config.model_name,
                prompt_tuning_init_text=config.prompt_init_text
            )
            model = get_prompt_tuning_model(
                model, peft_config=peft_config, adapter_name=f'{language}_prompt')
            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{language}_prompt')
    elif training_type == 'task':
        if language_adapter_type == 'adapter' and task_adapter_type == 'adapter':
            adapter_config = AdapterConfig.load(
                f'{config.language_adapter}/adapter_config.json')

            lang_adapter_name = model.load_adapter(
                f'{config.language_adapter}', config=adapter_config, load_as=f'{language}_adapter')

            model.add_adapter(f'{task_name}_adapter')
            model.add_seq2seq_lm_head(f'{task_name}_adapter')

            model.train_adapter([f'{task_name}_adapter'])
            model.set_active_adapters(
                [lang_adapter_name, f'{task_name}_adapter'])
        elif language_adapter_type == 'adapter' and task_adapter_type == 'prompt':
            adapter_config = AdapterConfig.load(
                f'{config.language_adapter}/adapter_config.json')

            lang_adapter_name = model.load_adapter(
                f'{config.language_adapter}', config=adapter_config, load_as=f'{language}_adapter')

            peft_task_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(config),
                num_virtual_tokens=config.num_virtual_tokens,
                tokenizer_name_or_path=config.model_name,
                prompt_tuning_init_text=config.prompt_init_text
            )

            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{task_name}_prompt')
            model.set_active_adapters(f'{config.language}_adapter')
            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{task_name}_prompt')
        elif language_adapter_type == 'prompt' and task_adapter_type == 'adapter':
            peft_lang_config = PromptTuningConfig.from_pretrained(
                f'{config.language_adapter}')

            config = AutoConfig.from_pretrained(
                peft_lang_config.base_model_name_or_path)
            model = AutoAdapterModel.from_pretrained(
                peft_lang_config.base_model_name_or_path, config=config)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, f'{config.language_adapter}')

            model.add_adapter(f'{task_name}_adapter')
            model.train_adapter(f'{task_name}_adapter')
            model.set_active_adapters(f'{task_name}_adapter')
        elif language_adapter_type == 'prompt' and task_adapter_type == 'prompt':
            peft_lang_config = PromptTuningConfig.from_pretrained(
                f'{config.language_adapter}')

            peft_task_config = PromptTuningConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init=get_promptinit(config),
                num_virtual_tokens=config.num_virtual_tokens,
                tokenizer_name_or_path=config.model_name,
                prompt_tuning_init_text=config.prompt_init_text,
                task_prompt=True
            )

            config = AutoConfig.from_pretrained(
                peft_lang_config.base_model_name_or_path)
            model = AutoAdapterModel.from_pretrained(
                peft_lang_config.base_model_name_or_path, config=config)

            model = PromptTuningForSeq2SeqLM.from_pretrained(
                model, f'{config.language_adapter}', adapter_name=f'{language}_prompt')

            model = get_prompt_tuning_model(
                model, peft_config=peft_task_config, adapter_name=f'{task_name}_prompt')

            model = freeze_parameters(model)
            unfreeze_parameters(model, f'{task_name}_prompt')
    else:
        raise ValueError(f"Invalid training type: {training_type}")

    return model, tokenizer


if __name__ == '__main__':
    args = get_args()

    wandb_config = get_wandb_config("../configs/api.conf")
    huggingface_config = get_huggingface_config("../configs/api.conf")
    # config = get_config(args.config_path)
    config = Config.from_yaml(args.config_path)

    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR
    os.environ['HF_API_KEY'] = huggingface_config.HF_API_KEY
    # os.environ['WANDB_MODE'] = 'offline'

    dataset = TaskDataset(
        config.model_id,
        config.task_name,
        config.template_name
    )
    max_token_length = dataset.get_max_target_length(
        config.tokenizer, config.max_target_length)

    model, tokenizer = get_model_tokenizer(
        args=args,
        model_name_or_path=config.model_name,
        config=config,
        task_name=config.task_name
    )

    train_data, valid_data, test_data = dataset.tokenize_dataset(
        tokenizer, 128, 128)

    metrics = dataset.get_metrics()

    train_data = all_mixing([train_data])
    valid_data = proportional_mixing(valid_data, round(524_288 * 0.2))
    logging.info(f"Train data size: {len(train_data)}")
    logging.info(f"Valid data size: {len(valid_data)}")

    if 'bert' in config.model_name:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    else:
        data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_data, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size)

    eval_dataloader = DataLoader(
        valid_data, collate_fn=data_collator, batch_size=config.batch_size)

    trainer = CustomTrainer(
        config=config,
        wandb_project=args.wandb_project,
        wandb_log_model=args.wandb_model,
        use_hf=args.use_hf,
        train_type=get_train_type(args),
        training=args.training,
        language_adapter=args.language_adapter,
        task_adapter=args.task_adapter,
    )

    trainer.train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        dataset_name=config.task_name,
    )
