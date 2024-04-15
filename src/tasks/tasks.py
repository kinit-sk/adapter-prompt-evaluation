from typing import Any
from tasks.dataset import Dataset
from tasks.utils import convert_language
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
import datasets
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
import numpy as np
import pandas as pd
import json

import logging

logging.basicConfig(level=logging.INFO)


class MLQA(Dataset):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'english'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)

    def load(self) -> None:
        if self.language == 'english':
            self.dataset = load_dataset('squad')
        else:
            train_dataset = load_dataset(
                'mlqa', name=f'mlqa-translate-train.{self.language}', split='train')

            valid_dataset = load_dataset(
                'mlqa', name=f'mlqa-translate-train.{self.language}', split='validation')

            test_dataset = load_dataset(
                'mlqa', name=f'mlqa-translate-test.{self.language}', split='test')

            self.dataset = DatasetDict(
                {'train': train_dataset, 'validation': valid_dataset, 'test': test_dataset})

    def _convert2squad(self, data):
        rows = []

        for record in data['data']:
            title = record['title']
            for paragraph in record['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    id = qa['id']
                    answers = qa['answers']
                    if len(answers) == 0:
                        answers = {"text": [], "answer_start": []}
                    elif len(answers) > 0:
                        answers = {
                            "text": [answer['text'] for answer in answers],
                            "answer_start": [answer['answer_start'] for answer in answers]
                        }
                    rows.append({'title': title, 'context': context,
                                'question': question, 'id': id, 'answers': answers})

        df = pd.DataFrame(rows)

        return df

    def preprocess(self, examples):
        questions = examples['question']
        contexts = examples['context']
        answers = examples['answers']

        def generate_input(_question, _context):
            return " ".join(["question:", _question.lstrip(), "context:", _context.lstrip()])

        inputs = [generate_input(question, context)
                  for question, context in zip(questions, contexts)]
        targets = [answer["text"][0] if len(
            answer["text"]) > 0 else "" for answer in answers]
        return inputs, targets

    def post_processing_function(
        self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval", tokenizer: Any = None
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {
            example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']}
                      for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


class SlovakSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'slovak'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def load(self):
        with open('../data/SKSQuAD/sk-quad-220614-train.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/SKSQuAD/sk-quad-220614-dev.json', 'r') as f:
            valid_data = json.load(f)

        train_data = self._convert2squad(train_data)
        valid_data = self._convert2squad(valid_data)

        train = HFDataset.from_pandas(train_data)
        valid = HFDataset.from_pandas(valid_data)

        valid = valid.filter(lambda x: len(x['answers']['text']) > 0)

        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': valid

        })


class CSSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'czech'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'validation',
        }

    def load(self):
        with open('../data/CSSQuAD/squad-2.0-cs/train-v2.0.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/CSSQuAD/squad-2.0-cs/dev-v2.0.json', 'r') as f:
            valid_data = json.load(f)

        train_data = self._convert2squad(train_data)
        valid_data = self._convert2squad(valid_data)

        train = HFDataset.from_pandas(train_data)
        valid = HFDataset.from_pandas(valid_data)

        valid = valid.filter(lambda x: len(x['answers']['text']) > 0)

        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': valid

        })
