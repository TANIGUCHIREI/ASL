
import transformers
from baseline.asl.customized_cache import DynamicCache
transformers.cache_utils.DynamicCache = DynamicCache # ...for time/allocation memory measuring

def get_llama_model(model_name_or_path, **kwargs):
    from baseline.pyramidinfer.models.modeling_llama_pyramidinfer import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model
def get_qwen_model(model_name_or_path, **kwargs):
    from baseline.pyramidinfer.models.modeling_qwen2_pyramidinfer import Qwen2ForCausalLM
    model = Qwen2ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return model

def get_model(model_name_or_path, **kwargs):
    
    if "Llama" in model_name_or_path:
        # model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        model = get_llama_model(model_name_or_path, **kwargs)
    elif "Qwen" in model_name_or_path:
        model = get_qwen_model(model_name_or_path, **kwargs)
    else:
        raise ValueError(f"{model_name_or_path=} is invalid in pyramidinfer.")
    return model

def load_pyramid_config(model, config):
    prefill_config = config['prefill_stage']
    for k, v in prefill_config.items():
        setattr(model.config, k, v)
        
    generation_config = config['generation_stage']
    for k, v in generation_config.items():
        setattr(model.config, k, v)
        
    return model