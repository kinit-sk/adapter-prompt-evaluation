from typing import Union
import yaml


class Config:
    learning_rate: float = 0.0001
    optimizer: str = 'AdamW'
    batch_size: int = 32
    epochs: int = 5
    weight_decay: float = 0.00001
    training_steps: int = 250_000
    eval_steps: int = 2_500
    save_steps: int = 2_500
    warm_init: bool = True
    num_warmup_steps: int = 0
    gradient_steps = 100
    fp16: bool = True
    model_name: str = 'bigscience/mt0-base'
    output_path: str = '../results'
    language: str = 'english'
    prompt_init_text: str = 'translate English to French: '
    eval_strategy: str = 'steps'
    padding: str = 'max_length'
    truncation: Union[str | bool] = True
    init_type: str = 'random'
    num_virtual_tokens: int = 50
    max_length: int = 128
    language_adapter: str = None
    task_adapter: str = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    # load from the path, that will read yaml file
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        return cls(**config)

    # convert to dict
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(k)}
