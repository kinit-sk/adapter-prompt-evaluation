import evaluate
from datasets import load_dataset
from typing import Any, Dict, List


class Dataset:
    def __init__(
        self,
            benchmark_name: str,
            subset: str = None,
            split: str = None,
            path: str = None
    ) -> None:
        self.benchmark_name = benchmark_name
        self.subset = subset
        self.split = split
        self.path = path
        self.dataset = None
        self.splits = {
            "train": "train",
            "validation": "validation",
            "test": "test"
        }
        self.load()

    def load(self) -> None:
        if self.subset is None:
            self.dataset = load_dataset(self.benchmark_name)
        else:
            self.dataset = load_dataset(self.benchmark_name, self.subset)

    def preprocess(self, examples):
        return NotImplementedError

    def tokenize(
            self,
            examples: Dict,
            tokenizer: Any,
            data_args: Any,
            max_seq_length: int = 128
    ):
        inputs, targets = self.preprocess(examples)
        padding = "max_length" if data_args.pad_to_max_length else False
        max_answer_length = data_args.max_answer_length

        model_inputs = tokenizer(
            inputs, max_length=max_seq_length, padding=padding, truncation=True)
        # Tokenize targets with text_target=...
        labels = tokenizer(
            text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_validation_function(
            self,
            examples: Dict,
            tokenizer: Any,
            data_args: Any,
            max_seq_length: int = 128

    ):
        inputs, targets = self.preprocess(examples)
        padding = "max_length" if data_args.pad_to_max_length else False
        max_answer_length = data_args.max_answer_length

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(
            text_target=targets, max_length=max_answer_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = model_inputs.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        model_inputs["example_id"] = []
        # Augment the overflowing tokens to the labels
        labels_out = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
            labels_out.append(labels["input_ids"][sample_index])

        model_inputs["labels"] = labels_out
        return model_inputs

    def post_processing_function(self, examples, features, outputs, stage, tokenizer):
        raise NotImplementedError

    def get_columns(self, split: str):
        return self.dataset[self.splits[split]].column_names

    def get_dataset(self, split: str):
        return self.dataset[self.splits[split]]

    def get_metric(self, metric_name: str):
        return evaluate.load(metric_name)
