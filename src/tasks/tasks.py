from typing import Any
from tasktemplates.dataset import Dataset

import logging

logging.basicConfig(level=logging.INFO)


class TaskDataset:
    def __init__(self, model_id: str, dataset_name: str, prompt_name: str) -> None:
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name

        self.dataset = Dataset(dataset_name, model_id, prompt_name)
        self.splits = self.dataset.splits
        self.label_names = self.dataset.prompt_template.choices
        self.metrics = self.dataset.prompt_template.get_metrics()

    def get_metrics(self):
        return self.metrics

    def tokenize_dataset(
            self,
            tokenizer: Any,
            max_input_length: int = 128,
            max_target_length: int = 128,
            padding: str = 'max_length',
            truncation: bool = True,
            pad_token: int = -100
    ):
        return self.dataset.tokenize_dataset(
            tokenizer,
            max_input_length,
            max_target_length,
            padding,
            truncation,
            pad_token
        )

    def preprocess_dataset(self, batched: bool = True, remove_columns: bool = True):
        self.dataset.preprocess(batched, remove_columns)

    def get_dataset(self):
        if not self.dataset.preprocessed:
            self.preprocess_dataset()
        return self.dataset.dataset

    def get_max_target_length(self, tokenizer, default_max_length):
        if self.label_names is not None:
            return max([len(tokenizer.encode(label)) for label in self.label_names])
        return default_max_length
