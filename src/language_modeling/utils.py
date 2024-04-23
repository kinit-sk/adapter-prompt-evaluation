
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration, MT5ForConditionalGeneration


def get_model(model_args, config, clm=False, t5_modeling=False):
    model_name_or_path = model_args.model_name_or_path
    if clm:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            # token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            # torch_dtype=model_args.dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    elif t5_modeling:
        if 'mt5' in model_args.model_name_or_path or 'mt0' in model_args.model_name_or_path:
            model = MT5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                # seed=seed,
                # dtype=model_args.dtype,
                # token=model_args.token,
            )
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                # seed=seed,
                # dtype=model_args.dtype,
                # token=model_args.token,
            )
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            # use_auth_token=True if model_args.use_auth_token else None,
        )

    return model
