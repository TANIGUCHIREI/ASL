from importlib.metadata import version
import warnings
import transformers
from baseline.asl.customized_cache import DynamicCache
transformers.cache_utils.DynamicCache = DynamicCache
import transformers

from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

from transformers.models.mistral.modeling_mistral import MISTRAL_ATTENTION_CLASSES

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version


def replace_llama():
    transformers_version = check_version()
    version_list = ['4.45']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with asl. asl is tested with Transformers version {version_list}.")
    from baseline.asl.llama_hijack import (
    LlamaFastKVAttention,
    llama_decoderlayer_forward,
    prepare_inputs_for_generation
)          
    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaFastKVAttention
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = llama_decoderlayer_forward
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
    
    
def replace_qwen2():
    transformers_version = check_version()
    version_list = ['4.45']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with asl. asl is tested with Transformers version {version_list}.")

    from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES
    from baseline.asl.qwen2_hjack import (
    Qwen2FlashAttention2,
    qwen2_decoderlayer_forward,
    prepare_inputs_for_generation
)
    QWEN2_ATTENTION_CLASSES['flash_attention_2'] = Qwen2FlashAttention2
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = qwen2_decoderlayer_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation

def replace_phi3():
    transformers_version = check_version()
    version_list = ['4.45']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with asl. asl is tested with Transformers version {version_list}.")

    from transformers.models.phi3.modeling_phi3 import PHI3_ATTENTION_CLASSES
    from baseline.asl.phi3_hjack import (
        Phi3FlashAttention2,
        phi3_decoderlayer_forward,
        prepare_inputs_for_generation,
)
    PHI3_ATTENTION_CLASSES['flash_attention_2'] = Phi3FlashAttention2
    transformers.models.phi3.modeling_phi3.Phi3DecoderLayer.forward = phi3_decoderlayer_forward
    transformers.models.phi3.modeling_phi3.Phi3ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation