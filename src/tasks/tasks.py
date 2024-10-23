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
import os
import re

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
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {
            example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        
        for example_index, example in enumerate(examples):
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


class ChineseSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        language: str = 'chinese'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def load(self) -> None:
        with open('../data/ChineseSQuAD/train-v1.1-zh.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/ChineseSQuAD/dev-v1.1-zh.json', 'r') as f:
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


class XQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'xquad',
        subset: str = None,
        split: str = None,
        language: str = 'greek'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def load(self) -> None:
        data = load_dataset('google/xquad', f'xquad.{self.language}', split='validation')
        
        self.dataset = DatasetDict({
            'train': None,
            'validation': None,
            'test': data
        })


class KenSwQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        language: str = 'swahili'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def load(self) -> None:
        data = load_dataset('lightblue/KenSwQuAD', split='train')
        data = data.map(lambda x: {'answers': {'text': x['answers']['text'], 'answer_start': []}})
        
        data = data.remove_columns('Story_ID')
        ids = [f'{i}' for i in range(len(data))]
        data = data.add_column('id', ids)
        
        data = data.train_test_split(test_size=0.15, seed=42)
        test = data['test']
        train = data['train'].train_test_split(test_size=0.15, seed=42)
        valid = train['test']
        train = train['train']
        
        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': test
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


class SlovenianSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'squad',
        subset: str = None,
        split: str = None,
        language: str = 'slovenian'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
    
    def _convert2squad(self, data):            
        df = pd.DataFrame(data['data'])
        return df
        
    def load(self) -> None:
        with open('../data/SlovenianSQuAD/squad2-slo-mt-train.json', 'r') as f:
            train_data = json.load(f)

        with open('../data/SlovenianSQuAD/squad2-slo-mt-dev.json', 'r') as f:
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
        

class IndicQA(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'indic_qa',
        subset: str = None,
        split: str = None,
        language: str = 'malaayalam'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def load(self) -> None:
        data = load_dataset('ai4bharat/IndicQA', f'indicqa.ml', split='test')
        
        self.dataset = DatasetDict({
            'train': None,
            'validation': None,
            'test': data
        })


class UQA(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'uqa',
        subset: str = None,
        split: str = None,
        language: str = 'urdu'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def _convert2squad(self, data):
        rows = []
        for row in data:
            index = row['id']
            if any(d['id'] == index for d in rows):
                for d in rows:
                    if d['id'] == index:
                        d['answers']['text'].append(row['answer'])
                        d['answers']['answer_start'].append(row['answer_start'])
            else:
                text = row['answer']
                answer_start = row['answer_start']
                answers = {'text': [text], 'answer_start': [answer_start]}
                rows.append({
                    'context': row['context'],
                    'question': row['question'],
                    'answers': answers,
                    'id': index
                })
            
        df = pd.DataFrame(rows)
        return df
    
        
    def load(self) -> None:
        train = load_dataset('uqa/UQA', split='train')
        test = load_dataset('uqa/UQA', split='validation')
        
        train = self._convert2squad(train)
        test = self._convert2squad(test)
        
        train = HFDataset.from_pandas(train).train_test_split(test_size=0.15, seed=42)
        valid = train['test']
        train = train['train']
        
        test = HFDataset.from_pandas(test)
        
        valid = valid.filter(lambda x: len(x['answers']['text']) > 0)
        test = test.filter(lambda x: len(x['answers']['text']) > 0)
        
        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': test
        })


class ArabicSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'arabic_squad',
        subset: str = None,
        split: str = None,
        language: str = 'arabic'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def _convert2squad(self, data):
        rows = []
        for row in data:
            index = row['index']
            text = row['text']
            answer_start = row['answer_start']
            answers = {'text': [text], 'answer_start': [answer_start]}
            rows.append({
                'context': row['context'],
                'question': row['question'],
                'answers': answers,
                'id': index
            })
            
        df = pd.DataFrame(rows)
        return df
        
    def load(self):
        data = load_dataset('Mostafa3zazi/Arabic_SQuAD', split='train')
        data = self._convert2squad(data)
        data = HFDataset.from_pandas(data)
        
        data = data.train_test_split(test_size=0.15, seed=42)
        test = data['test']
        train = data['train'].train_test_split(test_size=0.15, seed=42)
        valid = train['test']
        train = train['train']
        
        self.dataset = DatasetDict({
            'train': train,
            'validation': valid,
            'test': test
        })
        
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
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {
            example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}

        for example_index, example in enumerate(examples):
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        formatted_predictions = [
            {"id": k, "prediction_text": '--' if v == '' else v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']}
                      for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def get_metric(self):
        return evaluate.load('squad')
    
    
class SberSQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'sber_squad',
        subset: str = None,
        split: str = None,
        language: str = 'russian'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def load(self) -> None:
        train = load_dataset('kuznetsoffandrey/sberquad', split='train')
        validation = load_dataset('kuznetsoffandrey/sberquad', split='validation')
        test = load_dataset('kuznetsoffandrey/sberquad', split='test')
        
        train = train.cast_column('id', datasets.Value("string"))
        validation = validation.cast_column('id', datasets.Value("string"))
        test = test.cast_column('id', datasets.Value("string"))
        
        self.dataset = DatasetDict({
            'train': train,
            'validation': validation,
            'test': test
        })
    
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
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {
            example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}

        for example_index, example in enumerate(examples):
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        formatted_predictions = [
            {"id": k, "prediction_text": '--' if v == '' else v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']}
                      for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def get_metric(self):
        return evaluate.load('squad')
                     

class TeQuAD(MLQA):
    def __init__(
        self,
        benchmark_name: str = 'tequad',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'telugu'
    ) -> None:
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.splits = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }
        
    def _convert2squad(self, contexts, questions, answers, spans):
        rows = []
        for idx, context, question, answer, span in zip(range(len(contexts)), contexts, questions, answers, spans):
            rows.append({
                'context': context,
                'question': question,
                'answers': {'text': [answer.strip()], 'answer_start': [int(span.split('\t')[0])]},
                'id': str(idx)
            })
        
        df = pd.DataFrame(rows)
        return df

    def load(self) -> None:
        with open('../data/TeQuAD/Train/real_ans_tel.txt', 'r') as f:
            real_ans_train = f.readlines()
            
        with open('../data/TeQuAD/Train/real_que_tel.txt', 'r') as f:
            real_ques_train = f.readlines()
            
        with open('../data/TeQuAD/Train/real_con_tel.txt', 'r') as f:
            real_con_train = f.readlines()
            
        with open('../data/TeQuAD/Train/real_span_tel.txt', 'r') as f:
            real_span_train = f.readlines()
            
        with open('../data/TeQuAD/Test/Corrected/corrected_ans_tel.txt', 'r') as f:
            real_ans_test = f.readlines()
            
        with open('../data/TeQuAD/Test/Corrected/que_tel.txt', 'r') as f:
            real_ques_test = f.readlines()
            
        with open('../data/TeQuAD/Test/Corrected/real_con_tel.txt', 'r') as f:
            real_con_test = f.readlines()
            
        with open('../data/TeQuAD/Test/Corrected/span_tel.txt', 'r') as f:
            real_span_test = f.readlines()

        train_data = self._convert2squad(real_con_train, real_ques_train, real_ans_train, real_span_train)
        test_data = self._convert2squad(real_con_test, real_ques_test, real_ans_test, real_span_test)

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
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
            
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {
            example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}

        for example_index, example in enumerate(examples):
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        formatted_predictions = [
            {"id": k, "prediction_text": '--' if v == '' else v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']}
                      for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def get_metric(self):
        return evaluate.load('squad')


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
        self.splits = {
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
        self.label_names = ['Yes', 'Maybe', 'No'] #3
        # self.label_names = ['entailment', 'neutral', 'contradiction'] #4
        # self.label_names = ['True', 'Neither', 'False'] #3
        # self.label_names = ['true', 'inconclusive', 'false'] #4
        # self.label_names = ['always', 'sometimes', 'never'] #3
        # self.label_names = ['guaranteed', 'possible', 'impossible'] #4
        # self.label_names = ['correct', 'inconclusive', 'incorrect'] #3
        self.split_to_data_split = {
            'train': 'train',
            'validation': 'validation',
            'test': 'test',
        }

    def load(self):
        if self.language == 'cs':
            self.dataset = load_dataset('ctu-aic/anli_cs')
            self.dataset = self.dataset.rename_column('claim', 'hypothesis')
            self.dataset = self.dataset.rename_column('evidence', 'premise')
        elif self.language == 'sk':
            df_train = pd.read_csv('../data/SKANLI/anli_sk_train.csv')
            df_dev = pd.read_csv('../data/SKANLI/anli_sk_validation.csv')
            df_test = pd.read_csv('../data/SKANLI/anli_sk_test.csv')
            
            df_train = df_train.drop(columns=['translated'])
            df_dev = df_dev.drop(columns=['translated'])
            df_test = df_test.drop(columns=['translated'])
            
            df_train = df_train.rename(columns={'claim': 'hypothesis', 'evidence': 'premise'})
            df_dev = df_dev.rename(columns={'claim': 'hypothesis', 'evidence': 'premise'})
            df_test = df_test.rename(columns={'claim': 'hypothesis', 'evidence': 'premise'})
            
            self.dataset = DatasetDict({
                'train': HFDataset.from_pandas(df_train),
                'validation': HFDataset.from_pandas(df_dev),
                'test': HFDataset.from_pandas(df_test)
            })
        elif self.language == 'te' or self.language == 'ml':
            self.dataset = load_dataset('Divyanshu/indicxnli', name=f'{self.language}')
        elif self.language == 'ro':
            with open('../data/RONLI/train.json', 'r') as f:
                train_data = json.load(f)

            with open('../data/RONLI/validation.json', 'r') as f:
                valid_data = json.load(f)

            with open('../data/RONLI/test.json', 'r') as f:
                test_data = json.load(f)
                
            train_df = pd.DataFrame(train_data)
            valid_df = pd.DataFrame(valid_data)
            test_df = pd.DataFrame(test_data)
            
            train_df = train_df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
            valid_df = valid_df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
            test_df = test_df.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis'})
            
            train_df['label'] = train_df['label'].map({0: 2, 1: 0, 2: -1, 3: 1})
            valid_df['label'] = valid_df['label'].map({0: 2, 1: 0, 2: -1, 3: 1})
            test_df['label'] = test_df['label'].map({0: 2, 1: 0, 2: -1, 3: 1})
            
            train_df = train_df[train_df['label'] != -1]
            valid_df = valid_df[valid_df['label'] != -1]
            test_df = test_df[test_df['label'] != -1]
            
            self.dataset = DatasetDict({
                'train': HFDataset.from_pandas(train_df),
                'validation': HFDataset.from_pandas(valid_df),
                'test': HFDataset.from_pandas(test_df)
            })
        elif self.language == 'sl':
            data = load_dataset('cjvt/si_nli', 'public')
            data = data.map(lambda x: {'label': 0 if x['label'] == 'entailment' else 1 if x['label'] == 'neutral' else 2})
            self.dataset = data
        else:
            self.dataset = load_dataset('xnli', name=f'{self.language}')

    def preprocess(self, examples):
        premises = examples['premise']
        hypotheses = examples['hypothesis']
        labels = examples['label']
        
        def generate_input(_premise, _hypothesis):
            return f'{_premise} \n\nQuestion: Does this imply that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'{_premise}\n\nQuestion: {_hypothesis} True, False, or Neither?'
            # return f'Take the following as truth: {_premise}\n\nThen the following statement: "{_hypothesis}" is true, false, or inconclusive?'
            # return f'Given that {_premise} Does it follow that {_hypothesis} Yes, no, or maybe?'
            # return f'{_premise} Based on the previous passage, is it true that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'Given {_premise} Is it guaranteed true that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'Given {_premise} Should we assume that "{_hypothesis}" is true? Yes, no, or maybe?'
            # return f'Given that {_premise} Therefore, it must be true that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'Suppose {_premise} Can we infer that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'{_premise} Are we justified in saying that "{_hypothesis}"? Yes, no, or maybe?'
            # return f'{_premise} Based on that information, is the claim: "{_hypothesis}" true, false, or inconclusive?'
            # return f'Assume it is true that {_premise} \n\nTherefore, "{_hypothesis}" is guaranteed, possible, or impossible?'
            # return f'{_premise} Using only the above description and what you know about the world, "{_hypothesis}" is definitely correct, incorrect, or inconclusive?'
            
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
        label_names = [label.lower() for label in self.label_names]
        return [label_names.index(text.lower()) if text.lower() in label_names else 1 for text in texts]
    
    def post_processing_function(
        self, examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval", tokenizer: Any = None
    ):
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


class MultiClaimCheckWorthy(Dataset):
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
        self.label_names = ['Not checkworthy', 'Checkworthy']
        self.splits = {
            'train': 'train',
            'validation': 'valid',
            'test': 'test',
        }

    def load(self):
        if os.path.exists('../data/MultiClaim/train-all.csv'):
            train = pd.read_csv('../data/MultiClaim/train-all.csv')
            dev = pd.read_csv('../data/MultiClaim/dev-all.csv')
            test = pd.read_csv('../data/MultiClaim/test-all.csv')
        else:
            monat_data = pd.read_csv(
                f'../data/monant/MultiClaim.csv', sep=';')
            synthetic_data = pd.read_csv(
                f'../data/MultiClaim/synthetic.csv')

            datasets = pd.concat([monat_data, synthetic_data])
            datasets = datasets.sample(frac=1).reset_index(drop=True)
            split_ratio = [0.7, 0.15]
            idx = [int(datasets.shape[0] * split_ratio[i]) for i in range(2)]

            train = datasets.iloc[:idx[0]].copy()
            dev = datasets.iloc[idx[0]: idx[0] + idx[1]].copy()
            test = datasets.iloc[idx[0] + idx[1]:, :].copy()

            train.to_csv('../data/MultiClaim/train.csv', index=False)
            dev.to_csv('../data/MultiClaim/dev.csv', index=False)
            test.to_csv('../data/MultiClaim/test.csv', index=False)

        train['claim'] = train['claim'].astype(int)
        dev['claim'] = dev['claim'].astype(int)
        test['claim'] = test['claim'].astype(int)

        # get based on language
        train = train[[self.language, 'claim']]
        dev = dev[[self.language, 'claim']]
        test = test[[self.language, 'claim']]

        self.dataset = DatasetDict({
            'train': HFDataset.from_pandas(train),
            'valid': HFDataset.from_pandas(dev),
            'test': HFDataset.from_pandas(test)
        })
    
    def preprocess(self, examples):
        texts = examples[self.language]
        claims = examples['claim']
        
        def generate_input(_text):
            _text = re.sub(r'http\S+', '', _text)
            _text = _text.replace('\n', ' ')
            _text = _text.replace('\t', ' ')
            
            return f'checkworthiness claim: {_text}'

        inputs = [generate_input(text) for text in texts]
        targets = [self.label_names[claim] for claim in claims]

        return inputs, targets
    
    def convert_label(self, texts):
        return [self.label_names.index(text) if text in self.label_names else 0 if 'No' in text else 1 for text in texts]
    
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
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
            
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = self.convert_label(decoded_preds)
        
        labels = outputs.label_ids
                
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = self.convert_label(decoded_labels)

        return EvalPrediction(predictions=decoded_preds, label_ids=decoded_labels)

    def get_metric(self):
        return evaluate.load('f1')
    
    
class FEVERNLI(Dataset):
    def __init__(
        self, 
        benchmark_name: str = 'fever',
        subset: str = None,
        split: str = None,
        path: str = None,
        language: str = 'english'
        ):
        self.language = convert_language(language)
        super().__init__(benchmark_name, subset, split, path)
        self.label_names = ['No', 'Maybe', 'Yes']
        self.splits = {
            'train': 'train',
            'validation': 'dev',
            'test': 'test',
        }
        
    def load(self) -> None:
        if self.language == 'en':
            self.dataset = load_dataset('ctu-aic/enfever_nli')
        elif self.language == 'cs':
            self.dataset = load_dataset('ctu-aic/csfever_nli')
    
    def preprocess(self, examples):
        premises = examples['evidence']
        hypotheses = examples['claim']
        labels = examples['label']
        
        def generate_input(_premise, _hypothesis):
            return f'{_premise} \n\nQuestion: Does this imply that \"{_hypothesis}\"? Yes, no, or maybe?'

        inputs = [generate_input(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        targets = [self.label_names[label] for label in labels]
        return inputs, targets
    
    def convert_label(self, texts):
        return [self.label_names.index(text) for text in texts]
    
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
        return evaluate.load('F1')