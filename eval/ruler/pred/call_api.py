# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import time
import torch
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import traceback
from utils import load_data
import logging
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


SERVER_TYPES = (
    'trtllm',
    'vllm',
    'openai',
    'gemini',
    'hf',
    'mamba',
)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values



def get_output(args,model,tokenizer, outputs_parallel, idx, index, input:str, outputs, others, truncation, length):
    while True:
        try:

            #inifnite-benchから流用する
            input_tokens = tokenizer.encode(input)
            input_tensors = {
                    "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
                }
            input_length = len(input_tokens)
            # set args
            model.cache_kwargs = args
            if "gemfilter" in args.method:
                from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
                set_topk(model, args.select_k, mode='gemfilter')
                pred,prefill_time,output_time,output_length = gemfilter_generate_selection(
                    input_tensors["input_ids"], None, model, tokenizer, select_layer_idx=args.default_select_layer)
            else:
                pred = model.generate(**input_tensors,generation_config=args.generation_config)
                pred = pred[0, len(input_tokens) :]
                output_length = pred.shape[0]
                pred = tokenizer.decode(pred, skip_special_tokens=True)
                pred = pred.strip()
            break
        except Exception as e:
            traceback.print_exc()

    if len(pred) > 0:
        data = {
                    "id": index,
                    "prediction": pred,
                    "ground_truth": outputs,
                    'others': others,
                    'truncation': truncation,
                    'input': None,
                } #inputを入れてしまうとjsonlが非常に大きくなってしまうため
        analysis_data = {}
        if "gemfilter" in args.method:
            analysis_data["prefill_time"] = prefill_time
            analysis_data["output_time"] = output_time
            analysis_data["output_time"] = output_time
            analysis_data["input_length"] = input_length
            analysis_data["output_length"] = output_length
            allocated=0
            for i in range(torch.cuda.device_count()):
                allocated += torch.cuda.memory_allocated(i) #allocated容量を追加する
            analysis_data["allocated_memory"] = allocated
            analysis_data["select_layer"] = model.gemfilter_select_layer
        else:
            analysis_data =model.model.past_key_values.get_analysis_data()
            analysis_data["output_length"] = output_length
        data |=analysis_data
        outputs_parallel[idx] = data

def main(args):
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])
    max_new_len = config['tokens_to_generate']
    args.max_new_len = max_new_len

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        
        pred_index = [sample['id'] for sample in load_data(pred_file)]
        data = [sample for sample in load_data(task_file) if sample['index'] not in pred_index]
    else:
        data = load_data(task_file)

    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16

    #monkeypatch
    if args.method=="asl" or args.method=="fastkv":
        from baseline.asl.monkeypatch import replace_llama,replace_qwen2,replace_phi3
        replace_llama()
        replace_qwen2()
        replace_phi3()

        # Load Model & Tokenizer
        logging.info(f'Load Model & Tokenizer...')
        from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig, GenerationConfig 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        config = AutoConfig.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device, attn_implementation='flash_attention_2', torch_dtype=dtype) # , config=config
        model.eval()
    elif args.method=="pyramidinfer":
        import transformers
        from baseline.asl.customized_cache import DynamicCache
        transformers.cache_utils.DynamicCache = DynamicCache
        from baseline.pyramidinfer.utils import get_model, load_pyramid_config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if  "Qwen2.5-14B" in args.model_name:
            config_file = "qwen2.5_14b.json"
        elif  "Qwen2.5-7B" in args.model_name:
            config_file = "qwen2.5_7b.json"
        elif  "Llama-3.1-8B" in args.model_name:
            config_file = "llama3.1_8b.json"
        else:
            ValueError(f"model {args.model_name} has no config file")
        config_path = os.path.normpath(os.path.join(script_dir, f"../../../baseline/pyramidinfer/configs/{config_file}"))
        pyramid_model = get_model(
                args.model_name,
                torch_dtype=dtype,
                device_map=args.device,
                attn_implementation="eager",
                cache_dir=None,
                load_in_8bit=True if '70' in args.model_name or '34' in args.model_name else False,
            )
        pyramid_model.eval()
        print("Pyramidinfer Model GPU Memory Per GPU (MB): ", f"{torch.cuda.max_memory_allocated(device=pyramid_model.device) / 1024 / 1024:.3f}")
        # pyramid_model = torch.compile(pyramid_model, mode="max-autotune")
        from transformers import AutoTokenizer,GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map=args.device, trust_remote_code=True)
        pyramid_model.config.pad_token_id = tokenizer.pad_token_id
        pyramid_config = json.load(open(config_path))

        model = load_pyramid_config(pyramid_model, pyramid_config)

    elif args.method == 'gemfilter':
        import transformers
        from baseline.asl.customized_cache import DynamicCache
        transformers.cache_utils.DynamicCache = DynamicCache 
        from baseline.gemfilter.monkeypatch import replace_llama,replace_qwen2
        replace_llama()
        replace_qwen2()
        from transformers import AutoTokenizer,GenerationConfig,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map=args.device, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map=args.device, attn_implementation='flash_attention_2', torch_dtype=dtype)
        model.eval()
    elif args.method=="vanilla":
        import transformers
        from baseline.asl.customized_cache import DynamicCache
        transformers.cache_utils.DynamicCache = DynamicCache 
        from baseline.vanilla.monkeypatch import get_model
        model = get_model(
                args.model_name,
                torch_dtype=dtype,
                device_map=args.device,
                attn_implementation="flash_attention_2",
            )
        model.eval()
        from transformers import AutoTokenizer,GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map=args.device, trust_remote_code=True)
    else:
        ValueError(f"we dont support method: {args.method}")


    print("Model and tokenizer loaded.")

    args.generation_config = GenerationConfig(
        max_new_tokens=max_new_len,
        num_return_sequences=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]
    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    args=args,
                    model=model,
                    tokenizer=tokenizer,
                    outputs_parallel=outputs_parallel,
                    idx=idx,
                    index=data_point['index'],
                    input=data_point['input'],
                    outputs=data_point['outputs'],
                    others=data_point.get('others', {}),
                    truncation=data_point.get('truncation', -1),
                    length=data_point.get('length', -1),
                ),
            )
            thread.start()
            threads.append(thread)
            if len(threads) == args.threads:
                for thread in threads:
                    thread.join()
                threads = []
                for computed_idx in range(idx - args.threads + 1, idx + 1):
                    if len(outputs_parallel[computed_idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')

        # collecting the final batch
        if len(data) > 0:
            for thread in threads:
                thread.join()
            for computed_idx in range(idx - len(threads) + 1, idx + 1):
                if len(outputs_parallel[computed_idx]) > 0:
                    fout.write(json.dumps(outputs_parallel[computed_idx]) + '\n')
    
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
    parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
    parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
    parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
    parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

    # Server
    parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
    parser.add_argument("--server_host", type=str, default='127.0.0.1')
    parser.add_argument("--server_port", type=str, default='5000')
    parser.add_argument("--ssh_server", type=str)
    parser.add_argument("--ssh_key_path", type=str)

    # Inference
    parser.add_argument("--model_name", type=str, help="huggingface model name")
    parser.add_argument("--max_len", type=int, default=128000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="fp16",choices=["fp16", "bf16"])
    parser.add_argument("--threads", type=int, default=4)

    parser.add_argument("--synthetic_len", type=int, required=True)

    #For all methods
    parser.add_argument("--method", type=str, default="asl", choices=["vanilla","asl", "fastkv", "gemfilter", "pyramidinfer"])
    # Adaptive Select Layer
    def none_or_type(type_func):
        def convert(val):
            if val in ("None", "", None):
                return None
            return type_func(val)
        return convert
    parser.add_argument("--default_select_layer", type=none_or_type(int), default=None)
    parser.add_argument("--layer_th", type=none_or_type(float), default=None)
    parser.add_argument("--select_k", type=none_or_type(int), default=None)
    parser.add_argument("--window_size", type=none_or_type(int), default=None)
    parser.add_argument("--kernel_size", type=none_or_type(int), default=None)
    parser.add_argument("--use_layer_num", type=none_or_type(int), default=None)
    parser.add_argument("--pooling", type=none_or_type(str), default=None)
    parser.add_argument("--cache_k", type=none_or_type(int), default=None)
    parser.add_argument("--cache_p", type=none_or_type(float), default=None)

    #pyramidinfer
    # parser.add_argument("--cache_dir", type=str, default=None)
    # parser.add_argument("--max_new_tokens", type=int, default=None)
    # parser.add_argument("--pyramid_bsz", type=int, default=32)
    # parser.add_argument("--original_bsz", type=int, default=32)
    # parser.add_argument("--pyramid_config", type=str, default="configs/llama2_7b.json")
    # parser.add_argument("--pyramid_enable", action="store_false")
    # parser.add_argument("--original_enable", action="store_false")
    # parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    print(args)

    if args.server_type == 'hf' or args.server_type == 'gemini':
        args.threads = 1

    seed_everything(42)
    main(args)