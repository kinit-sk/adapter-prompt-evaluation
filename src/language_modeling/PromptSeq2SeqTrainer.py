import importlib.metadata
import importlib.util
import os
import torch
from torch import nn
from transformers import Trainer, PreTrainedModel
from typing import Optional, Union, Tuple
import logging

from prompt_tuning import PromptTuningForSeq2SeqLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SAFE_WEIGHTS_NAME = "model.safetensors"
TRAINING_ARGS_NAME = "training_args.bin"
WEIGHTS_NAME = "pytorch_model.bin"


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_peft_available = _is_package_available("peft")
_safetensors_available = _is_package_available("safetensors")


def is_peft_available():
    return _peft_available


def is_safetensors_available():
    return _safetensors_available


if is_safetensors_available():
    import safetensors.torch


if is_peft_available():
    from peft import PeftModel


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class PromptSeq2SeqTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            f"PromptSeq2SeqTrainer::Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (
            PreTrainedModel, PeftModel, PromptTuningForSeq2SeqLM)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(
                        output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
