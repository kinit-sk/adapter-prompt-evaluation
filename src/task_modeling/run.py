import logging
import os
import sys

import adapters
from functools import partial
from adapters import AdapterArguments
from task_modeling.trainer_seq2seq_qa import QuestionAnsweringSeq2SeqAdapterTrainer, QuestionAnsweringSeq2SeqTrainer, QuestionAnsweringSeq2SeqAdapterTrainerWithPrompt, QuestionAnsweringSeq2SeqTrainerWithPrompt
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from language_modeling.args import PromptTuningArguments
from transformers.trainer_utils import EvalPrediction
from task_modeling.args import ModelArguments, DataTrainingArguments
from task_modeling.utils import get_model, get_updated_model
from tasks import dataset_factory


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterArguments, PromptTuningArguments))
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

    dataset = dataset_factory(
        dataset_name=data_args.dataset_name, language=prompt_args.language)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = get_model(model_args, config)

    # Convert the model into an adapter model
    if not model_args.full_finetuning:
        adapters.init(model)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to generate and tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset.get_columns('train')
    elif training_args.do_eval:
        column_names = dataset.get_columns('validation')
    elif training_args.do_predict:
        column_names = dataset.get_columns('test')
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    preprocess_function = partial(
        dataset.tokenize,
        tokenizer=tokenizer,
        data_args=data_args,
        max_seq_length=max_seq_length
    )

    preprocess_validation_function = partial(
        dataset.preprocess_validation_function,
        tokenizer=tokenizer,
        data_args=data_args,
        max_seq_length=max_seq_length
    )

    if training_args.do_train:
        train_dataset = dataset.get_dataset("train")
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_examples = dataset.get_dataset("validation")
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(
                len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        predict_examples = dataset.get_dataset("test")
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(
                range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = dataset.get_metric(
        "squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    post_processing_function = partial(
        dataset.post_processing_function,
        tokenizer=tokenizer,
    )

    # Setup adapters
    # if adapter_args.train_adapter:
    #     setup_adapter_training(model, adapter_args,
    #                            data_args.dataset_name or "mlm")
    if not model_args.full_finetuning:
        model = get_updated_model(
            model, model_args, adapter_args, prompt_args, data_args.dataset_name or "mlm")

    # Initialize our Trainer
    if adapter_args.train_adapter and model_args.load_language_prompt is not None:
        trainer_class = QuestionAnsweringSeq2SeqAdapterTrainerWithPrompt
    elif prompt_args.prompt_tuning:
        trainer_class = QuestionAnsweringSeq2SeqTrainerWithPrompt
    else:
        trainer_class = (
            QuestionAnsweringSeq2SeqAdapterTrainer if adapter_args.train_adapter else QuestionAnsweringSeq2SeqTrainer
        )

    print(model)
    print(trainer_class)

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        post_process_function=post_processing_function,
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
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_answer_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "question-answering"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
