from typing import Any, Dict
from collections import namedtuple
from metrics.metrics import span_f1
from tasks.dataset import Dataset
from tasks.utils import convert_language
from datasets import load_dataset, DatasetDict
from datasets import Dataset as HFDataset
import datasets
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
import numpy as np
import pandas as pd
import json
import evaluate

import logging

logging.basicConfig(level=logging.INFO)

Metric = namedtuple('Metric', ['name', 'compute'])


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
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }

    def load(self) -> None:
        if self.language == 'en':
            train_dataset = load_dataset(
                'squad', split='train')
            # split train_dataset into train and validation
            train_dataset = train_dataset.train_test_split(
                test_size=0.15, seed=42)
            
            valid_dataset = train_dataset['test']
            train_dataset = train_dataset['train']
            
            test_dataset = load_dataset(
                'squad', split='validation')
            
            self.dataset = DatasetDict(
                {'train': train_dataset, 'validation': valid_dataset, 'test': test_dataset})
        else:
            train_dataset = load_dataset(
                'mlqa', name=f'mlqa-translate-train.{self.language}', split='train')
            
            # split train_dataset into train and validation
            train_dataset = train_dataset.train_test_split(
                test_size=0.15, seed=42)
            valid_dataset = train_dataset['test']
            train_dataset = train_dataset['train']

            test_dataset = load_dataset(
                'mlqa', name=f'mlqa-translate-train.{self.language}', split='validation')

            self.dataset = DatasetDict(
                {'train': train_dataset, 'validation': valid_dataset, 'test': test_dataset})
        
        print(self.dataset)

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

    def get_metric(self):
        return evaluate.load('squad')

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
            'test': 'test',
        }

    def load(self):
        with open('../data/SKSQuAD/sk-quad-220614-train.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/SKSQuAD/sk-quad-220614-dev.json', 'r') as f:
            test_data = json.load(f)

        train_data = self._convert2squad(train_data)
        test_data = self._convert2squad(test_data)

        train = HFDataset.from_pandas(train_data).train_test_split(test_size=0.15, seed=42)
        valid = train['test']
        train = train['train']
        
        test = HFDataset.from_pandas(test_data)

        valid = valid.filter(lambda x: len(x['answers']['text']) > 0)
        test = test.filter(lambda x: len(x['answers']['text']) > 0)

        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': test
        })
        print(self.dataset)


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
            'test': 'test',
        }

    def load(self):
        with open('../data/CSSQuAD/squad-2.0-cs/train-v2.0.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/CSSQuAD/squad-2.0-cs/dev-v2.0.json', 'r') as f:
            test_data = json.load(f)

        train_data = self._convert2squad(train_data)
        test_data = self._convert2squad(test_data)

        train = HFDataset.from_pandas(train_data).train_test_split(test_size=0.15, seed=42)
        valid = train['test']
        train = train['train']
        
        test = HFDataset.from_pandas(test_data)

        test = test.filter(lambda x: len(x['answers']['text']) > 0)
        valid = valid.filter(lambda x: len(x['answers']['text']) > 0)

        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': test
        })
        print(self.dataset)


class WikiANN(Dataset):
    def __init__(
        self, 
        benchmark_name: str = 'wikiann',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'english'
    ):
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        

    def load(self):
        self.dataset = load_dataset('wikiann', name=f'{self.language}')

    def preprocess(self, examples):
        tokens = examples['tokens']
        spans = examples['spans']
        
        def generate_input(_tokens):
            # return f'Sentence: {" ".join(_tokens)}\nIdentify all named entities in the sentence using PER, LOC, ORG.'
            return f'tag: {" ".join(_tokens)}'

        inputs = [generate_input(token) for token in tokens]
        # targets = [', '.join(span) for span in spans]
        targets = ['$$'.join(span) for span in spans]

        return inputs, targets
    
    def preprocess_validation_function(
            self,
            examples: Dict,
            tokenizer: Any,
            data_args: Any,
            max_seq_length: int = 128

    ):
        return self.tokenize(examples, tokenizer, data_args, max_seq_length)
        
    def post_processing_function(
        self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval", tokenizer: Any = None
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = outputs.label_ids
                
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        return EvalPrediction(predictions=decoded_preds, label_ids=decoded_labels)
    
    def get_metric(self):
        return Metric(name='span_f1', compute=span_f1)
    
    
class XNLI(Dataset):
    def __init__(
        self,
        benchmark_name: str = 'xnli',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'english'
    ):
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.label_names = ['Yes', 'Maybe', 'No']
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }

    def load(self):
        self.dataset = load_dataset('xnli', name=f'{self.language}')

    def preprocess(self, examples):
        premises = examples['premise']
        hypotheses = examples['hypothesis']
        labels = examples['label']
        
        def generate_input(_premise, _hypothesis):
            return f'{_premise} \n\nQuestion: Does this imply that \"{_hypothesis}\"? Yes, no, or maybe?'

        inputs = [generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        targets = [self.label_names[label] for label in labels]

        return inputs, targets
    
    def preprocess_validation_function(
            self,
            examples: Dict,
            tokenizer: Any,
            data_args: Any,
            max_seq_length: int = 128

    ):
        return self.tokenize(examples, tokenizer, data_args, max_seq_length)
    
    def convert_label(self, texts):
        return [self.label_names.index(text) for text in texts]
    
    def post_processing_function(
        self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval", tokenizer: Any = None
    ):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        predictions = self.convert_label(decoded_preds)
        
        labels = outputs.label_ids
                
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        references = self.convert_label(decoded_labels)

        return EvalPrediction(predictions=predictions, label_ids=references)
    
    def get_metric(self):
        return evaluate.load('xnli')
