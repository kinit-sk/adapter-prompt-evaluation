import torch
import os

WEIGHTS_NAME = 'adapter_model.bin'


def _prepare_prompt_learning_config(prompt_config, model_config):
    if prompt_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `prompt_config`")
        prompt_config.num_layers = num_layers

    if prompt_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        prompt_config.token_dim = token_dim

    if prompt_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError(
                "Please specify `num_attention_heads` in `peft_config`")
        prompt_config.num_attention_heads = num_attention_heads

    if getattr(prompt_config, "encoder_hidden_size", None) is None:
        setattr(prompt_config, "encoder_hidden_size", prompt_config.token_dim)

    return prompt_config


def _get_batch_size(input_ids, inputs_embeds):
    if input_ids is not None:
        return input_ids.shape[0]
    elif inputs_embeds is not None:
        return inputs_embeds.shape[0]
    else:
        return None


def get_peft_model_state_dict(model, state_dict=None, adapter_name='default'):
    config = model._peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    to_return = {}
    if config.inference_mode:
        prompt_embeddings = model.prompt_encoder[adapter_name].embeddings.weight
    else:
        prompt_embeddings = model.get_prompt_embedding_to_save(
            adapter_name)
    to_return["prompt_embeddings"] = prompt_embeddings
    to_return = {k.replace(f".{adapter_name}", "")                 : v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name='default'):
    state_dict = {}
    if getattr(model, 'modules_to_save', None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f'{module_name}.modules_to_save.{adapter_name}')
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    peft_model_state_dict = state_dict
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    model.prompt_encoder[adapter_name].embeddings.load_state_dict(
        {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
    )
    return load_result


def infer_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_adapter_weights(model_id, device=None):
    if device is None:
        device = infer_device()

    adapter_weights = torch.load(os.path.join(
        model_id, WEIGHTS_NAME), map_location=device)
    return adapter_weights
