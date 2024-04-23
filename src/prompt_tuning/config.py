import os
from huggingface_hub import hf_hub_download
from dataclasses import asdict, dataclass
import json
import inspect
from transformers.utils import PushToHubMixin
import enum

CONFIG_NAME = 'adapter_config.json'


class PromptTuningInit(str, enum.Enum):
    TEXT = 'TEXT'
    RANDOM = 'RANDOM'
    SAMPLED = 'SAMPLED'
    CLASS = 'CLASS'


class TaskType(str, enum.Enum):
    SEQ_2_SEQ_LM = 'SEQ_2_SEQ_LM'
    CAUSAL_LM = 'CAUSAL_LM'


@dataclass
class PromptTuningConfig(PushToHubMixin):

    auto_mapping: dict = None
    base_model_name_or_path: str = None
    revision: str = None
    task_type: str = TaskType.SEQ_2_SEQ_LM
    inference_mode: bool = False
    num_virtual_tokens: int = 20
    token_dim: int = None
    num_transformer_submodules: int = None
    num_layers: int = None
    prompt_tuning_init: str = PromptTuningInit.RANDOM
    prompt_tuning_init_text: str = None
    num_attention_heads: int = None
    tokenizer_name_or_path: str = 't5-base'
    peft_type: str = 'PROMPT_TUNING'
    task_prompt: bool = False
    fusion: str = 'none' # 'cat', 'avg', 'none'

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception:
                raise ValueError(
                    f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'")

        loaded_attributes = cls.from_json_file(config_file)
        config_cls = cls

        kwargs = {**class_kwargs, **loaded_attributes}
        return config_cls(**kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError(
                "Provided path ({}) should be a directory, not a file".format(save_directory))

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop('auto_mapping_dict', None)

        output_dict = asdict(self)
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)
        if auto_mapping_dict is not None:
            output_dict['auto_mapping'] = auto_mapping_dict

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(output_dict, ensure_ascii=False,
                    indent=2, sort_keys=True))

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs
