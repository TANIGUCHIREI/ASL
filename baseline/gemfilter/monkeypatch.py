from importlib.metadata import version
import warnings
import transformers
from baseline.gemfilter.llama_hijack_4_45 import LlamaSelectAttention, llama_model_forward
from baseline.gemfilter.qwen2_hijack_4_45 import Qwen2SelectAttention, qwen2_model_forward
from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES
# from baseline.gemfilter.mistral_hijack_4_45 import MistralSelectAttention, mistral_model_forward
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
            f"Transformers version {transformers_version} might not be compatible with FastKV. FastKV is tested with Transformers version {version_list}.")

    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaSelectAttention
    transformers.models.llama.modeling_llama.LlamaModel.forward = llama_model_forward
    from baseline.asl.llama_hijack import prepare_inputs_for_generation
    
    transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation #[als] for time analysis
    from baseline.vanilla.modeling_llama import LlamaDecoderLayer
    transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = LlamaDecoderLayer.forward #for mlp chunk calcuation

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
            f"Transformers version {transformers_version} might not be compatible with FastKV. FastKV is tested with Transformers version {version_list}.")

    QWEN2_ATTENTION_CLASSES['flash_attention_2'] = Qwen2SelectAttention
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_model_forward
    from baseline.asl.llama_hijack import prepare_inputs_for_generation
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation  #[als] for time analysis
    from baseline.vanilla.modeling_qwen2 import Qwen2DecoderLayer
    transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = Qwen2DecoderLayer.forward #for mlp chunk calcuation
    
# def replace_mistral():
#     transformers_version = check_version()
#     version_list = ['4.45']
#     warning_flag = True
#     for version in version_list:
#         if version in transformers_version:
#             warning_flag = False
#             break
#     if warning_flag:
#         warnings.warn(
#             f"Transformers version {transformers_version} might not be compatible with FastKV. FastKV is tested with Transformers version {version_list}.")

#     MISTRAL_ATTENTION_CLASSES['flash_attention_2'] = MistralSelectAttention
#     transformers.models.mistral.modeling_mistral.MistralModel.forward = mistral_model_forward

