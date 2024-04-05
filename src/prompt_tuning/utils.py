import copy
import inspect
import torch
import os
from contextlib import nullcontext
import accelerate
from accelerate.hooks import add_hook_to_module, remove_hook_from_module

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


class ModulesToSaveWrapper(torch.nn.Module):
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_module()

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        # Try to anticipate some modules that users could try to target that would not work.
        # Note: It's not possible to check hasattr(module, "forward"), since that returns True for ModuleDict and
        # ModuleList, even though their forward methods cannot be called
        forbidden_classes = (torch.nn.ModuleDict, torch.nn.ModuleList,
                             torch.nn.ParameterDict, torch.nn.ParameterList)
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__.__name__
            raise TypeError(
                f"modules_to_save cannot be applied to modules of type {cls_name}")

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def weight(self):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module.weight
        return self.modules_to_save[self.active_adapter].weight

    def update(self, adapter_name):
        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                import deepspeed

                context_manager = deepspeed.zero.GatheredParameters(
                    self.original_module.parameters(), modifier_rank=0)
                break
        with context_manager:
            self.modules_to_save.update(torch.nn.ModuleDict(
                {adapter_name: copy.deepcopy(self.original_module)}))

        if hasattr(self.modules_to_save[adapter_name], "_hf_hook"):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)

        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        r"""
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def forward(self, *args, **kwargs):
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save):
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(
                f"Adapter {adapter_name} not found in {self.modules_to_save.keys()}")

        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key)
                                  for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)
