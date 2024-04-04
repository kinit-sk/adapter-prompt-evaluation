import torch
from transformers import AutoTokenizer
import math
from prompt_tuning.config import PromptTuningInit
from train.initialization import init_tokens, class_initialization


class PromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()
        init_type = config.prompt_tuning_init

        total_virtual_tokens = config.num_virtual_tokens * \
            config.num_transformer_submodules
        self.embeddings = torch.nn.Embedding(
            total_virtual_tokens, config.token_dim)
        
        if init_type == PromptTuningInit.TEXT:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_tokens_ids = tokenizer(init_text)['input_ids']
            num_tokens = len(init_tokens_ids)
            if num_tokens > total_virtual_tokens:
                init_tokens_ids = init_tokens_ids[:total_virtual_tokens]
            elif num_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_tokens)
                init_tokens_ids = init_tokens_ids * num_reps

            init_tokens_ids = init_tokens_ids[:total_virtual_tokens]
            init_tokens_ids = torch.LongTensor(
                init_tokens_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                init_tokens_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embeddings.weight = torch.nn.Parameter(word_embedding_weights)

        elif init_type == PromptTuningInit.SAMPLED:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            sampled_tokens = init_tokens(tokenizer, total_virtual_tokens)
            num_tokens = len(sampled_tokens)
            if num_tokens > total_virtual_tokens:
                sampled_tokens = sampled_tokens[:total_virtual_tokens]
            elif num_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_tokens)
                sampled_tokens = sampled_tokens * num_reps

            sampled_tokens = sampled_tokens[:total_virtual_tokens]
            sampled_tokens = torch.LongTensor(
                sampled_tokens).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                sampled_tokens).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embeddings.weight = torch.nn.Parameter(word_embedding_weights)

        elif init_type == PromptTuningInit.CLASS:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            class_tokens = config.prompt_tuning_init_text
            classes = class_tokens.split(',')
            class_tokens = class_initialization(tokenizer, classes)
            num_tokens = len(class_tokens)
            if num_tokens > total_virtual_tokens:
                class_tokens = class_tokens[:total_virtual_tokens]
            elif num_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_tokens)
                class_tokens = class_tokens * num_reps

            class_tokens = class_tokens[:total_virtual_tokens]
            class_tokens = torch.LongTensor(
                class_tokens).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                class_tokens).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embeddings.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        return self.embeddings(indices)
