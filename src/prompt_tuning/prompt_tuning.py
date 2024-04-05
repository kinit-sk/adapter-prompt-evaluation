import torch
import os
from transformers import PreTrainedModel
from copy import deepcopy
import inspect
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
import logging

from prompt_tuning.config import PromptTuningConfig
from prompt_tuning.model import PromptEmbedding
from prompt_tuning.utils import _prepare_prompt_learning_config, _get_batch_size, get_peft_model_state_dict, infer_device, load_adapter_weights, set_peft_model_state_dict, _set_trainable

logging.basicConfig(level=logging.INFO)


WEIGHTS_NAME = 'adapter_model.bin'


class PeftModel(PushToHubMixin, torch.nn.Module):
    def __init__(self, model, peft_config, adapter_name='default'):
        super().__init__()
        # self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.peft_type = 'prompt_tuning'
        self.adapter_name = adapter_name
        self.device = model.device

        self._peft_config = {adapter_name: peft_config}
        self.base_model = model
        self.config = getattr(self.base_model, "config",
                              {"model_type": "custom"})
        self.add_adapter(adapter_name, peft_config)

        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise AssertionError(
                "Provided path ({}) should be a directory, not a file".format(save_directory))

        selected_adapters = list(self._peft_config.keys())
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name in selected_adapters:
            prompt_config = self._peft_config[adapter_name]
            output_state_dict = get_peft_model_state_dict(
                self, state_dict=kwargs.get("state_dict", None), adapter_name=adapter_name)

            output_dir = os.path.join(
                save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            torch.save(output_state_dict, os.path.join(
                output_dir, WEIGHTS_NAME))

            if prompt_config.base_model_name_or_path is None:
                prompt_config.base_model_name_or_path = (
                    self.base_model.__dict__.get('name_or_path', None)
                )

            inference_mode = prompt_config.inference_mode
            prompt_config.inference_mode = True

            auto_mapping_dict = None
            prompt_config.save_pretrained(
                output_dir, auto_mapping_dict=auto_mapping_dict)
            prompt_config.inference_mode = inference_mode

    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model, model_id, adapter_name='default', is_trainable=False, config=None, **kwargs):
        config = PromptTuningConfig.from_pretrained(model_id, **kwargs)

        model = cls(model, config, adapter_name=adapter_name)
        model.load_adapter(model_id, adapter_name=adapter_name,
                           is_trainable=is_trainable, **kwargs)
        return model

    def _setup_prompt_encoder(self, adapter_name):
        config = self._peft_config[adapter_name]

        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None

        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2

        for named_param, value in list(transformer_backbone.named_parameters()):
            deepspeed_distributed_tensor_shape = getattr(
                value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(
                    named_param.replace(".weight", ""))
                break

        prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        prompt_encoder.to(self.device)

        self.prompt_encoder.update(
            torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def get_prompt_embedding_to_save(self, adapter_name='default'):
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name]
            .unsqueeze(0)
            .expand(1, -1)
            .to(prompt_encoder.embeddings.weight.device)
        )

        prompt_embeddings = prompt_encoder(prompt_tokens)
        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size):
        prompt_config = self._peft_config[self.adapter_name]

        if list(self.prompt_encoder.keys()) == 1:
            prompt_encoder = self.prompt_encoder[self.adapter_name]

            prompt_tokens = (
                self.prompt_tokens[self.adapter_name]
                .unsqueeze(0)
                .expand(batch_size, -1)
                .to(prompt_encoder.embeddings.weight.device)
            )
            if prompt_config.inference_mode:
                return prompt_encoder.embeddings.weight.repeat(batch_size, 1, 1)
            else:
                return prompt_encoder(prompt_tokens)
        else:
            prompt_embeddings = []
            for adapter_name in self.prompt_encoder.keys():
                prompt_encoder = self.prompt_encoder[adapter_name]
                prompt_tokens = (
                    self.prompt_tokens[adapter_name]
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                    .to(prompt_encoder.embeddings.weight.device)
                )
                if prompt_config.inference_mode:
                    prompt_embeddings.append(
                        prompt_encoder.embeddings.weight.repeat(batch_size, 1, 1)[:, :prompt_config.num_virtual_tokens])
                else:
                    prompt_embeddings.append(prompt_encoder(prompt_tokens)[
                                             :, :prompt_config.num_virtual_tokens])

            if prompt_config.fusion == 'avg':
                prompt_embeddings = torch.stack(prompt_embeddings, dim=1)
                return torch.mean(prompt_embeddings, dim=1)
            elif prompt_config.fusion == 'cat':
                return torch.cat(prompt_embeddings, dim=1)
            else:
                raise ValueError(
                    f"Invalid fusion method: {prompt_config.fusion}")

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    @classmethod
    def _split_kwargs(cls, kwargs):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def get_base_model(self):
        return self.base_model

    def load_adapter(self, model_id, adapter_name='default', is_trainable=False, **kwargs):
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        adapter_weights = load_adapter_weights(
            model_id, device=torch_device, **hf_hub_download_kwargs)
        load_result = set_peft_model_state_dict(
            self, adapter_weights, adapter_name=adapter_name)
        if (
            (getattr(self, 'hf_device_map', None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({'cpu', 'disk'})) > 0)
            and len(self._peft_config) == 1
        ):
            device_map = kwargs.get('device_map', 'auto')
            max_memory = kwargs.get('max_memory', None)
            offload_dir = kwargs.get('offload_dir', None)
            offload_index = kwargs.get('offload_index', None)

            dispatch_model_kwargs = {}
            no_split_module_classes = self._no_split_modules

            if 'offload_index' in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs['offload_index'] = offload_index

            if device_map != 'sequential':
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == 'balanced_low_0')
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs
            )
            hook = AlignDevicesHook(io_same_device=True)
            remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        if not is_trainable:
            self.eval()
        return load_result

    def add_adapter(self, adapter_name, prompt_config):
        self._peft_config[adapter_name] = prompt_config
        if hasattr(self.config, "to_dict"):
            dict_config = self.config.to_dict()
        else:
            dict_config = self.config

        prompt_config = _prepare_prompt_learning_config(
            prompt_config, dict_config)
        self._setup_prompt_encoder(adapter_name)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(
            *args, **kwargs)

        return model_kwargs


class PromptTuningForSeq2SeqLM(PeftModel):

    def __init__(self, model, peft_config, adapter_name='default'):
        super().__init__(model, peft_config, adapter_name=adapter_name)

        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        self._prepare_encoder_decoder_kwargs_for_generation = self.base_model._prepare_encoder_decoder_kwargs_for_generation
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        prompt_config = self._peft_config[self.adapter_name]
        batch_size = _get_batch_size(input_ids, inputs_embeds)

        num_virtual_tokens = prompt_config.num_virtual_tokens
        if prompt_config.fusion == 'cat':
            num_virtual_tokens *= 2

        if decoder_attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, num_virtual_tokens).to(decoder_attention_mask.device)

        kwargs.update({
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        })

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens).to(
                attention_mask.device
            )
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat(
            (prompts[:, :num_virtual_tokens], inputs_embeds), dim=1)
        # logging.info(f"inputs_embeds: {inputs_embeds.shape}")

        return self.base_model(
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **kwargs
        )

    def generate(self, **kwargs):
        prompt_config = self._peft_config[self.adapter_name]
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model_prepare_encoder_decoder_kwargs_for_generation
        )
        try:
            if 'input_ids' not in kwargs and 'inputs_embeds' in kwargs:
                return self.base_model.generate(**kwargs)

            # kwargs['position_ids'] = None
            # kwargs['token_type_ids'] = None

            num_virtual_tokens = prompt_config.num_virtual_tokens
            if prompt_config.fusion == 'cat':
                num_virtual_tokens *= 2

            kwargs = deepcopy(kwargs)

            if 'encoder_outputs' in kwargs:
                del kwargs['encoder_outputs']

            input_ids = kwargs.pop('input_ids', None)
            inputs_embeds = self.word_embeddings(input_ids)
            batch_size = inputs_embeds.shape[0]
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)

            inputs_embeds = torch.cat(
                (prompts[:, :num_virtual_tokens], inputs_embeds), dim=1)
            kwargs['inputs_embeds'] = inputs_embeds

            if 'attention_mask' in kwargs:
                prefix_attention_mask = torch.ones(batch_size, num_virtual_tokens).to(
                    kwargs['attention_mask'].device
                )
                kwargs['attention_mask'] = torch.cat(
                    (prefix_attention_mask, kwargs['attention_mask']), dim=1)
            return self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
                self.base_model_prepare_encoder_decoder_kwargs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(
            *args, **kwargs)

        return model_kwargs


class PeftModelForCausalLM(PeftModel):
    def __init__(self, model, peft_config, adapter_name='default'):
        super().__init__(model, peft_config, adapter_name=adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        prompt_config = self._peft_config[self.adapter_name]
        batch_size = _get_batch_size(input_ids, inputs_embeds)

        num_virtual_tokens = prompt_config.num_virtual_tokens
        if prompt_config.fusion == 'cat':
            num_virtual_tokens *= 2

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)

        kwargs.update({
            'attention_mask': attention_mask,
            'labels': labels,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        })

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, num_virtual_tokens),
                -100
            ).to(labels.device)
            kwargs['labels'] = torch.cat(
                (prefix_labels, labels), dim=1)

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat(
            (prompts, inputs_embeds), dim=1)
        return self.base_model(
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def generate(self, **kwargs):
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, 'model'):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        prompt_config = self._peft_config[self.adapter_name]
        model_kwargs = self.base_model_prepare_inputs_for_generation(
            *args, **kwargs)

        num_virtual_tokens = prompt_config.num_virtual_tokens
        if prompt_config.fusion == 'cat':
            num_virtual_tokens *= 2

        if model_kwargs.get("attention_mask", None) is not None:
            size = model_kwargs["input_ids"].shape[0], num_virtual_tokens
            prefix_attention_mask = torch.ones(size).to(
                model_kwargs["input_ids"].device)
            model_kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            )

        if model_kwargs.get("position_ids", None) is not None:
            model_kwargs["position_ids"] = None

        if kwargs.get("token_type_ids", None) is not None:
            model_kwargs["token_type_ids"] = None

        if model_kwargs["past_key_values"] is None:
            inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
            prompts = self.get_prompt(
                batch_size=model_kwargs["input_ids"].shape[0])
            prompts = prompts.to(inputs_embeds.dtype)
            model_kwargs["inputs_embeds"] = torch.cat(
                (prompts, inputs_embeds), dim=1)
            model_kwargs["input_ids"] = None

        return model_kwargs


class PeftModelForQuestionAnswering(PeftModel):
    def __init__(self, model, peft_config, adapter_name='default'):
        super().__init__(model, peft_config, adapter_name=adapter_name)
        if self.modules_to_save is None:
            self.modules_to_save = {"qa_outputs"}
        else:
            self.modules_to_save.update({"qa_outputs"})

        for name, _ in self.base_model.named_children():
            if any(module_name in name for module_name in self.modules_to_save):
                self.cls_layer_name = name
                break

        # to make sure classifier layer is trainable
        _set_trainable(self, adapter_name)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        prompt_config = self._peft_config[self.adapter_name]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = _get_batch_size(input_ids, inputs_embeds)

        num_virtual_tokens = prompt_config.num_virtual_tokens
        if prompt_config.fusion == 'cat':
            num_virtual_tokens *= 2

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(
                batch_size, num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat(
                (prefix_attention_mask, attention_mask), dim=1)
        if kwargs.get("position_ids", None) is not None:
            kwargs["position_ids"] = None

        kwargs.update({
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        })

        if kwargs.get("token_type_ids", None) is not None:
            kwargs["token_type_ids"] = torch.cat(
                (
                    torch.zeros(batch_size, num_virtual_tokens).to(
                        self.word_embeddings.weight.device),
                    kwargs["token_type_ids"],
                ),
                dim=1,
            ).long()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        return self.base_model(inputs_embeds=inputs_embeds, **kwargs)
