from adapters import AdapterTrainer
import os
import torch
from typing import Any, List
from train.utils import create_arguments, get_optimizer
from transformers import Trainer, EarlyStoppingCallback, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
import logging

logging.basicConfig(level=logging.INFO)


class CustomTrainer:
    def __init__(
        self,
        config: Any,
        wandb_project: str = "wandb_project",
        wandb_log_model: str = "checkpoint",
        use_hf: bool = False,
        train_type: str = "prompt",  # prompt or adapter
        training: str = "task",  # task or language
        language_adapter: str = "prompt",  # prompt or adapter
        task_adapter: str = "prompt",  # prompt or adapter
    ) -> None:
        self.config = config
        self.wandb_project = wandb_project
        self.wandb_log_model = wandb_log_model
        self.use_hf = use_hf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_type = train_type
        self.training = training
        self.language_adapter = language_adapter
        self.task_adapter = task_adapter

    def _get_prompts(self, model: Any):
        if self.training == 'language' and self.language_adapter == 'prompt':
            return [
                model.get_prompt_embedding_to_save(
                    adapter_name=f'{self.config.language}_prompt')
            ]
        elif self.training == 'task':
            if self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
                return [
                    model.get_prompt_embedding_to_save(
                        adapter_name=f'{self.config.language}_prompt')
                ]
            elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
                return [
                    model.get_prompt_embedding_to_save(
                        adapter_name=f'{self.config.language}_prompt'),
                    model.get_prompt_embedding_to_save(
                        adapter_name=f'{self.task_name}_prompt')
                ]
            elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
                return [
                    model.get_prompt_embedding_to_save(
                        adapter_name=f'{self.task_name}_prompt')
                ]
            else:
                return None
        else:
            return None

    def _create_model_dir(self):
        os.makedirs(self.config.output_path, exist_ok=True)

        if self.training == 'language':
            os.makedirs(
                f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}', exist_ok=True)
            return f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}'
        else:
            os.makedirs(
                f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}/{self.task_name}_{self.task_adapter}', exist_ok=True)
            return f'{self.config.output_path}/{self.config.model_name.split("/")[-1]}/{self.config.language}_{self.language_adapter}/{self.task_name}_{self.task_adapter}'

    def _save_model(self, model: Any, tokenizer: Any, prompts: List[Any], total_steps: int, best: bool = False):
        logging.info('Saving model')
        config = self.config

        if best:
            path = f'{self._create_model_dir()}/best_model'
        else:
            # path = f'{config.output_path}/{config.model_name.split("/")[-1]}/language_{self.lanugage_adapter}/task_{self.task_adapter}/{dataloader.name}-{self.config["language"]}/checkpoint_{total_steps}'
            path = f'{self._create_model_dir()}/checkpoint_{total_steps}'

        model.save_pretrained(path)
        tokenizer.save_pretrained(path)

        if self.training == 'language':
            if self.language_adapter == 'adapter':
                model.save_adapter(
                    path, f'{config.language}_adapter', with_head=True)
            elif self.language_adapter == 'prompt':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
        else:
            if self.language_adapter == 'adapter' and self.task_adapter == 'adapter':
                model.save_adapter(
                    path, f'{config.language}_adapter', with_head=True)
                model.save_adapter(
                    path, f'{self.task_name}_adapter', with_head=True)
            elif self.language_adapter == 'adapter' and self.task_adapter == 'prompt':
                model.save_adapter(
                    path, f'{config.language}_adapter', with_head=True)
                torch.save(prompts[0], f'{path}/{self.task_name}.pt')
            elif self.language_adapter == 'prompt' and self.task_adapter == 'adapter':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
                model.save_adapter(
                    path, f'{self.task_name}_adapter', with_head=True)
            elif self.language_adapter == 'prompt' and self.task_adapter == 'prompt':
                torch.save(prompts[0], f'{path}/{config.language}.pt')
                torch.save(prompts[1], f'{path}/{self.task_name}.pt')

        return path

    def _train_hf(self, model: Any, train_dataset: Any, eval_dataset: Any):
        config = self.config

        training_args = create_arguments(len(train_dataset), config)

        optimizer = get_optimizer(config, model)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=(len(train_dataset) * config.epochs),
        )

        SelectedTrainer = Trainer if self.train_type == "prompt" else AdapterTrainer
        trainer = SelectedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()
        trainer.save_model(
            f"{config.output_path}{config.model_name.split('/')[-1]}-finetuned")

    def _train(self, model: Any, tokenizer: Any, train_dataset: Any, eval_dataset: Any, dataset_name: str):
        os.makedirs(config["output_path"], exist_ok=True)
        os.makedirs(
            f'{config.output_path}/{config.model_name.split("/")[-1]}-{dataset_name.name}-{config.language}', exist_ok=True)
        config = self.config

        optimizer = get_optimizer(config, model)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=(len(train_dataset) * config.epochs),
        )

        run = wandb.init(project=self.wandb_project, name=self.wandb_log_model)
        model = model.to(config.device)
        run.watch(model, log="all", log_freq=100)

        best_eval_loss = float("inf")
        total_steps = 0.0
        total_loss = 0.0

        print(torch.cuda.memory_summary())

        while True:
            model.train()

            for batch in tqdm(train_dataset):
                optimizer.zero_grad()
                batch = {k: v.to(config.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                total_steps += 1
                total_loss += loss.detach().float()

            if total_steps % config.eval_steps == 0 or total_steps == config.training_steps:
                wandb.log({"train_loss": total_loss / 100})
                total_loss = 0.0
                eval_loss = 0.0

                with torch.no_grad():
                    model.eval()
                    for batch in eval_dataset:
                        batch = {k: v.to(config.device)
                                 for k, v in batch.items()}
                        outputs = model(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                        eval_loss += outputs.loss.detach().float()

                total_eval_loss = eval_loss / len(eval_dataset)
                if total_steps == config.training_steps:
                    total_train_loss = total_loss / \
                        (total_steps % config.eval_steps)
                else:
                    total_train_loss = total_loss / config.eval_steps

                wandb.log({
                    "train_loss": total_train_loss,
                    "eval_loss": eval_loss
                }, commit=True)
                print(
                    f"Steps: {total_steps}, Train Loss: {total_train_loss}, Eval Loss: {total_eval_loss}.")

                prompts = self._get_prompts(model)
                if total_steps % self.config.save_steps == 0:
                    self._save_model(
                        model, tokenizer, prompts, total_steps)

                if total_eval_loss < best_eval_loss:
                    wandb.run.summary["best_eval_loss"] = total_eval_loss
                    best_eval_loss = total_eval_loss
                    best_path = self._save_model(
                        model, tokenizer, prompts, total_steps, best=True)
                    model_artifact = wandb.Artifact(
                        f"best-model",
                        type="model",
                        description="The best model so far.",
                        metadata=dict(config._asdict())
                    )
                    model_artifact.add_dir(best_path)
                    run.log_artifact(model_artifact)

                if total_steps == config.training_steps:
                    break

        run.finish()

    def train(self, model: Any, tokenizer: Any, train_dataset: Any, eval_dataset: Any, dataset_name: str = None):
        if self.use_hf:
            self._train_hf(model, train_dataset, eval_dataset)
        else:
            self._train(model, tokenizer, train_dataset,
                        eval_dataset, dataset_name)
