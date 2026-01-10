
def get_llama_model(model_name_or_path, **kwargs):
    from baseline.vanilla.modeling_llama import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model
def get_qwen_model(model_name_or_path, **kwargs):
    from baseline.vanilla.modeling_qwen2 import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model

def get_model(model_name_or_path, **kwargs):
    
    if "Llama" in model_name_or_path:
        # model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model = get_llama_model(model_name_or_path, **kwargs)
    elif "Qwen" in model_name_or_path:
        model = get_qwen_model(model_name_or_path, **kwargs)
    else:
        raise ValueError(f"{model_name_or_path=} is invalid in vanilla.")
    return model