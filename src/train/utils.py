import yaml
from transformers import TrainingArguments
from torch.optim import AdamW, Adamax, Adagrad, Adadelta, SparseAdam, Adam
from transformers import Adafactor


def create_arguments(num_samples, config):
    batch_size = config.batch_size
    model_name = config.model_name
    lr = config.learning_rate
    eval_strategy = config.eval_strategy
    gradient_accumulation_steps = config.gradient_steps
    epochs = config.epochs
    wd = config.weight_decay
    use_fp16 = config.fp16

    logging_steps = num_samples // (batch_size * epochs)
    # eval around each 2000 samples
    logging_steps = round(2000 / (batch_size * gradient_accumulation_steps))

    model_name = model_name.split("/")[-1]

    return TrainingArguments(
        output_dir=f"{config.output_path}{model_name}-finetuned",
        overwrite_output_dir=True,
        evaluation_strategy=eval_strategy,
        logging_steps=logging_steps,
        save_strategy=eval_strategy,
        save_steps=logging_steps,
        eval_steps=logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=epochs,
        weight_decay=wd,
        push_to_hub=False,
        do_train=True,
        do_eval=True,
        report_to="wandb",
        fp16=use_fp16,  # mdeberta not working with fp16
        warmup_steps=0,
        logging_dir=f"{config.output_path}{model_name}-finetuned/logs",
    )


def get_optimizer(config, model):

    optimizer = config.optimizer
    optimizer_lr = config.learning_rate
    wd = config.weight_decay

    if optimizer == 'Adadelta':
        return Adadelta(
            model.parameters(),
            lr=optimizer_lr
        )
    elif optimizer == 'Adagrad':
        return Adagrad(
            model.parameters(),
            lr=optimizer_lr
        )
    elif optimizer == 'Adam':
        return Adam(
            model.parameters(),
            lr=optimizer_lr
        )
    elif optimizer == 'AdamW':
        return AdamW(
            model.parameters(),
            lr=optimizer_lr,
            weight_decay=wd
        )
    elif optimizer == 'SparseAdam':
        return SparseAdam(
            model.parameters(),
            lr=optimizer_lr
        )
    elif optimizer == 'Adamax':
        return Adamax(
            model.parameters(),
            lr=optimizer_lr
        )
    elif optimizer == 'Adafactor':
        return Adafactor(
            model.parameters(),
            lr=optimizer_lr,
            scale_parameter=False,
            relative_step=False,
        )

    return Adam(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=wd
    )
