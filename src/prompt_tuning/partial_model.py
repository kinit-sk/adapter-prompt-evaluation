import torch
from transformers import AutoTokenizer
import math
from prompt_tuning.config import PromptTuningInit
from train.initialization import init_tokens, class_initialization


class CustomEmbedding(torch.nn.Module):
    weight: torch.Tensor
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.weight = torch.empty(self.num_embeddings, self.embedding_dim, device=device)
        torch.nn.init.normal_(self.weight)


class PartialPromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()
        self.config = config
        self.word_embeddings = word_embeddings
        init_type = config.prompt_tuning_init
        partial_init_text = config.partial_prompt_tuning_init_text
        self.num_fixed = config.fixed_size

        total_virtual_tokens = config.num_virtual_tokens * \
            config.num_transformer_submodules
        self.num_to_learn = total_virtual_tokens - self.num_fixed
        
        self.embeddings = CustomEmbedding(
            total_virtual_tokens, config.token_dim, device=word_embeddings.weight.device)
        
        # Fixed weights
        if partial_init_text is not None or partial_init_text != '':
            self.set_fixed(partial_init_text)
            
        # Trainable weights
        self.trainable_weight = torch.nn.Parameter(torch.empty(self.num_to_learn, config.token_dim))
        
        if init_type == PromptTuningInit.TEXT:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            init_text = config.prompt_tuning_init_text
            init_tokens_ids = tokenizer(init_text)['input_ids']
            num_tokens = len(init_tokens_ids)
            if num_tokens > self.num_to_learn:
                init_tokens_ids = init_tokens_ids[:self.num_to_learn]
            elif num_tokens < self.num_to_learn:
                num_reps = math.ceil(self.num_to_learn / num_tokens)
                init_tokens_ids = init_tokens_ids * num_reps

            init_tokens_ids = init_tokens_ids[:self.num_to_learn]
            init_tokens_ids = torch.LongTensor(
                init_tokens_ids).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                init_tokens_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.trainable_weight = torch.nn.Parameter(word_embedding_weights)

        elif init_type == PromptTuningInit.SAMPLED:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            sampled_tokens = init_tokens(tokenizer, self.num_to_learn)
            num_tokens = len(sampled_tokens)
            if num_tokens > self.num_to_learn:
                sampled_tokens = sampled_tokens[:self.num_to_learn]
            elif num_tokens < self.num_to_learn:
                num_reps = math.ceil(self.num_to_learn / num_tokens)
                sampled_tokens = sampled_tokens * num_reps

            sampled_tokens = sampled_tokens[:self.num_to_learn]
            sampled_tokens = torch.LongTensor(
                sampled_tokens).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                sampled_tokens).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.trainable_weight = torch.nn.Parameter(word_embedding_weights)

        elif init_type == PromptTuningInit.CLASS:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path)
            class_tokens = config.prompt_tuning_init_text
            classes = class_tokens.split(',')
            class_tokens = class_initialization(tokenizer, classes)
            num_tokens = len(class_tokens)
            if num_tokens > self.num_to_learn:
                class_tokens = class_tokens[:self.num_to_learn]
            elif num_tokens < self.num_to_learn:
                num_reps = math.ceil(self.num_to_learn / num_tokens)
                class_tokens = class_tokens * num_reps

            class_tokens = class_tokens[:self.num_to_learn]
            class_tokens = torch.LongTensor(
                class_tokens).to(word_embeddings.weight.device)

            word_embedding_weights = word_embeddings(
                class_tokens).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.trainable_weight = torch.nn.Parameter(word_embedding_weights)
        else:
            torch.nn.init.normal_(self.trainable_weight)
            
        self.embeddings.weight[self.num_fixed:] = self.trainable_weight

    def forward(self, indices):
        self.embeddings.weight.detach_()
        self.embeddings.weight[self.num_fixed:] = self.trainable_weight
        return torch.nn.functional.embedding(
            indices, self.embeddings.weight, None, None, 2., False, False)
        
    def set_fixed(self, fixed_text):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name_or_path)
        init_tokens_ids = tokenizer(fixed_text)['input_ids']
        num_tokens = len(init_tokens_ids)
        if num_tokens > self.num_fixed:
            init_tokens_ids = init_tokens_ids[:self.num_fixed]
        elif num_tokens < self.num_fixed:
            num_reps = math.ceil(self.num_fixed / num_tokens)
            init_tokens_ids = init_tokens_ids * num_reps

        init_tokens_ids = init_tokens_ids[:self.num_fixed]
        init_tokens_ids = torch.LongTensor(
            init_tokens_ids).to(self.word_embeddings.weight.device)
        word_embedding_weights = self.word_embeddings(
            init_tokens_ids).detach().clone()
        word_embedding_weights = word_embedding_weights.to(torch.float32)
        self.embeddings.weight[:self.num_fixed] = torch.nn.Parameter(word_embedding_weights)

    def to(self, device):
        self.embeddings.weight = self.embeddings.weight.to(device)
        # self.trainable_weight = self.trainable_weight.to(device)
        return self