# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations
import sys
sys.path.append(".")
import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple
import logging
import torch
import argparse

from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import set_seed


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))



# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tokenizer, max_tokens, manner: str = "middle"):
    tokens = tokenizer.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tokenizer, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tokenizer.decode(input_tokens[:200]))
        print("...")
        print(tokenizer.decode(input_tokens[-200:]))
        print("=====================================")

    input_tensors = {
        "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
    }
    input_length = len(input_tokens)
    prefill_time,output_time,output_length, = None,None,None
    # outputs = model.generate(**input_tensors, generation_config=generation_config)
    if "gemfilter" in args.method:
        from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
        set_topk(model, args.select_k, mode='gemfilter')
        output,prefill_time,output_time,output_length = gemfilter_generate_selection(
            input_tensors["input_ids"], None, model, tokenizer, select_layer_idx=args.default_select_layer)
    else:
        pred = model.generate(**input_tensors,generation_config=generation_config)
        pred = pred[0, len(input_tokens) :]
        output_length = pred.shape[0]
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        output = pred.strip()

    print("Chunked generation:", output)
    return output,prefill_time,output_time,output_length,input_length

def none_or_type(type_func):
    def convert(val):
        if val in ("None", "", None):
            return None
        return type_func(val)
    return convert


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    #Infinite-bench
    parser.add_argument(
        "--task",
        type=str,
        # choices=list(DATA_NAME_TO_MAX_NEW_TOKENS.keys()) + ["all"],
        required=True,
        help='Which task to use. Note that "all" can only be used in `compute_scores.py`.',  # noqa
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data", help="The directory of data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Where to dump the prediction results.",
    )  # noqa
    # parser.add_argument(
    #     "--model_name_or_path",
    #     type=str,
    #     default="facebook/opt-350m",
    #     help="For `compute_scores.py` only, specify which model you want to compute the score for.",  # noqa
    # )
    parser.add_argument(
        "--num_eval_examples",
        type=int,
        default=-1,
        help="The number of test examples to use, use all examples in default.",
    )  # noqa
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="The index of the first example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data.",
    )  # noqa
    parser.add_argument(
        "--stop_idx",
        type=int,
        help="The index of the last example to infer on. This is used if you want to evaluate on a (contiguous) subset of the data. Defaults to the length of dataset.",
    )  # noqa
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_sparq", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_seq_length", type=int, default=100000)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--start_example_id", type=int, default=0)


    # Model Arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    # parser.add_argument("--save_path", default="", type=str, help="Path to save the output")

    #For all methods
    parser.add_argument("--method", type=str, default="asl", choices=["vanilla","asl", "fastkv", "gemfilter", "pyramidinfer"])

    # Adaptive Select Layer
    parser.add_argument("--default_select_layer", type=none_or_type(int), default=None)
    parser.add_argument("--layer_th", type=none_or_type(float), default=None)
    parser.add_argument("--select_k", type=none_or_type(int), default=None)
    parser.add_argument("--window_size", type=none_or_type(int), default=None)
    parser.add_argument("--kernel_size", type=none_or_type(int), default=None)
    parser.add_argument("--use_layer_num", type=none_or_type(int), default=None)
    parser.add_argument("--pooling", type=none_or_type(str), default=None)
    parser.add_argument("--cache_k", type=none_or_type(float), default=None)
    parser.add_argument("--cache_p", type=none_or_type(float), default=None)

    # parser.add_argument("--mode", type=str, default="asl", choices=["asl"])

    #これ以降はKV compress用
    # parser.add_argument("--max_capacity_prompt", type=int, default=512)


    args = parser.parse_args()
    set_seed(args.seed)
    check_benchmark_availability(args.data_dir)
    
    from eval.ruler.decide_results_dir import decide_name
    real_model_name = decide_name(args)
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]


    dtype =torch.bfloat16

    # Load Model & Tokenizer
    logging.info(f'Load Model & Tokenizer...')

    # monkeypatch
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
        config_path = os.path.normpath(os.path.join(script_dir, f"../../baseline/pyramidinfer/configs/{config_file}"))
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
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
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
    from compute_scores import compute_scores

    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= args.max_seq_length:
            max_new_tokens = 500
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}")
        # print(F"あああ {args.output_dir=}, {real_model_name=}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        print(F"{args.rewrite=}")
        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, args.max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id or i < len(preds):
                continue
            input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print(f"====== Example {i} ======")

            if "Mistral-Nemo-Instruct-2407" in args.model_name:
                msgs = [dict(role="user", content=input_text)]
            else:
                msgs = [dict(role="system", content=input_text)]
            input_text = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )

            # set args
            model.cache_kwargs = args


            pred,prefill_time,output_time,output_length,input_length = get_pred(
                model,
                tokenizer,
                input_text,
                max_input_length=args.max_seq_length - max_new_tokens,
                generation_config=generation_config,
                verbose=args.verbose,
            )
            print("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )

            analysis_data = {}
            analysis_data = {}
            if "gemfilter" in args.method:
                analysis_data["prefill_time"] = prefill_time
                analysis_data["output_time"] = output_time
                analysis_data["output_time"] = output_time
                analysis_data["input_length"] = input_length
                analysis_data["output_length"] = output_length
                allocated=0
                for i in range(torch.cuda.device_count()):
                    allocated += torch.cuda.memory_allocated(i)
                analysis_data["allocated_memory"] = allocated
                analysis_data["select_layer"] = model.gemfilter_select_layer
            else:
                analysis_data =model.model.past_key_values.get_analysis_data()
                analysis_data =model.model.past_key_values.get_analysis_data()
                analysis_data["output_length"] = output_length
                analysis_data["input_length"] = input_length

            preds[-1] |=analysis_data
        
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))