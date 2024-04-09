import logging
import math
import os
import sys
from itertools import chain

from datasets import load_dataset, DatasetDict
from transformers.testing_utils import CaptureLogger

import adapters
import evaluate
import transformers
from adapters import AdapterArguments, AdapterTrainer, setup_adapter_training
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
    default_data_collator,
)
from prompt_tuning import PromptTuningConfig, TaskType, get_prompt_tuning_model
from utils import get_promptinit, freeze_parameters, unfreeze_parameters

from language_modeling.t5_mlm import compute_input_and_target_lengths, DataCollatorForT5MLM
from language_modeling.args import ModelArguments, DataTrainingArguments, PromptTuningArguments
from language_modeling.utils import get_model
from language_modeling.PromptSeq2SeqTrainer import PromptSeq2SeqTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments, PromptTuningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args, prompt_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args, prompt_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # TODO: Create custom Dataloader that will handle local datasets and also datasets from HF
    if data_args.dataset_name is not None:
        dataset_name = 'wikimedia/wikipedia'
        # dataset_name = 'wikitext'
        # Downloading and loading a dataset from the hub.
        raw_datasets = DatasetDict({
            "train": load_dataset(
                dataset_name,
                data_args.dataset_config_name,
                split=f"train[:1000000]",
                use_auth_token=True if model_args.use_auth_token else None,
            ),
            "validation": load_dataset(
                dataset_name,
                data_args.dataset_config_name,
                split="train[1000000:1100000]",
                use_auth_token=True if model_args.use_auth_token else None,
            )
        })
        # raw_datasets['train'] = load_dataset(
        #     dataset_name,
        #     data_args.dataset_config_name,
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         use_auth_token=True if model_args.use_auth_token else None,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         use_auth_token=True if model_args.use_auth_token else None,
        #     )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # TODO: Create custom Dataloader that will handle local datasets and also datasets from HF

    config_kwargs = {
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, **config_kwargs)

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = get_model(
            model_args=model_args,
            config=config,
            clm=data_args.clm,
            t5_modeling=data_args.t5_modeling
        )
    else:
        raise ValueError(
            "You are instantiating a new model from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --model_name_or_path."
        )

    # Convert the model into an adapter model
    adapters.init(model)

    if prompt_args.prompt_tuning:
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            prompt_tuning_init=get_promptinit(prompt_args.prompt_tuning_init),
            num_virtual_tokens=prompt_args.num_virtual_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
            prompt_tuning_init_text=prompt_args.prompt_tuning_init_text
        )

        model = get_prompt_tuning_model(
            model, peft_config=peft_config, adapter_name=f'{prompt_args.language}_prompt'
        )
        model = freeze_parameters(model)
        unfreeze_parameters(model, f'{prompt_args.language}_prompt')

        print(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        logger.info(
            f"The tokenizer picked has a vocab size of {len(tokenizer)}, but the model has an embedding size of {embedding_size}. "
            "You might want to consider resizing the embeddings or using a smaller tokenizer."
        )
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length,
                             tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        if data_args.clm:
            tok_logger = transformers.utils.logging.get_logger(
                "transformers.tokenization_utils_base")

            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                # clm input could be much much longer than block_size
                if "Token indices sequence length is longer than the" in cl.out:
                    tok_logger.warning(
                        "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                        " before being passed to the model."
                    )
                return output

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )

        elif not data_args.t5_modeling:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            with training_args.main_process_first(desc="dataset map tokenization"):
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
        elif data_args.t5_modeling:
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_attention_mask=False)

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            expanded_inputs_length, targets_length = compute_input_and_target_lengths(
                inputs_length=max_seq_length,
                noise_density=data_args.mlm_probability,
                mean_noise_span_length=data_args.mean_noise_span_length,
            )

            max_seq_length = expanded_inputs_length

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (
                    total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with training_args.main_process_first(desc="grouping texts together"):
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        if data_args.clm:
            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                labels = labels[:, 1:].reshape(-1)
                preds = preds[:, :-1].reshape(-1)
                return metric.compute(predictions=preds, references=labels)
        else:
            def compute_metrics(eval_preds):
                preds, labels = eval_preds
                labels = labels.reshape(-1)
                preds = preds.reshape(-1)
                mask = labels != -100
                labels = labels[mask]
                preds = preds[mask]
                return metric.compute(predictions=preds, references=labels)

    if data_args.t5_modeling:
        data_collator = DataCollatorForT5MLM(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_probability,
            mean_noise_span_length=data_args.mean_noise_span_length,
            input_length=data_args.max_seq_length,
            target_length=targets_length,
            pad_token_id=model.config.pad_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
    elif not data_args.clm:
        # Data collator
        # This one will take care of randomly masking the tokens.
        pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
    else:
        data_collator = default_data_collator

    if adapter_args.train_adapter:
        # Setup adapters
        setup_adapter_training(model, adapter_args,
                               data_args.dataset_name or "mlm")

    # Initialize our Trainer
    if prompt_args.prompt_tuning:
        trainer_class = AdapterTrainer if adapter_args.train_adapter else PromptSeq2SeqTrainer
    else:
        trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if data_args.clm:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "text-generation"}
    elif data_args.t5_modeling:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "t5-mlm"}
    else:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "fill-mask"}

    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
