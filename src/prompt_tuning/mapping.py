from prompt_tuning.utils import _prepare_prompt_learning_config
from prompt_tuning.prompt_tuning import PromptTuningForSeq2SeqLM, PeftModelForCausalLM

MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_2_SEQ_LM": PromptTuningForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM
}


def get_prompt_tuning_model(model, peft_config, adapter_name='default'):
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config = _prepare_prompt_learning_config(peft_config, model_config)

    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name)
