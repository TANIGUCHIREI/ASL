# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# copied and modified from https://github.com/microsoft/MInference/tree/main/experiments/needle_in_a_haystack
import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from needle_tools import LLMNeedleHaystackTester
from needle_viz import plot_needle_viz

# rulerからimportするための↓
import sys 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(PROJECT_ROOT)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from eval.ruler.decide_results_dir import decide_name

@dataclass
class Config:
    # wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl
    haystack_file: str = "data/pg19_mini.jsonl"  # Path to the haystack file
    model_name: str = "01-ai/Yi-9B-200K"
    run_name: str = None  # Name of the run, used for the output file
    context_lengths_min: int = 30_000
    context_lengths_max: int = 100_000
    n_context_length_intervals: int = 15  # Number of intervals between min and max ...default is 15
    n_document_depth_intervals: int = 10  # position of the needle in the haystack
    n_rounds: int = 5
    seed: int = 42
    output_path: str = "results/needle/"
    pattern_path: str = "config/Llama_3_8B_Instruct_262k_kv_out_v32_best_pattern.json"
    jobs: str = None
    trust_remote_code: bool = False
    args: dict = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.real_model_name = decide_name(self.args)
        if not os.path.exists(os.path.join(self.output_path, self.real_model_name)):
            os.makedirs(os.path.join(self.output_path, self.real_model_name))
        
        output_file = f"{self.real_model_name}/needle_res_{self.context_lengths_min}_{self.context_lengths_max}.json"
        # output_file = f"needle_res_{self.model_name.replace('/', '-')}_{self.run_name if self.run_name is not None else ''}_{self.jobs if self.jobs is not None else ''}_{timestamp}_{self.context_lengths_min}_{self.context_lengths_max}_{self.pattern_path.split('/')[-1].replace('.json', '') if self.pattern_path is not None else ''}.json"
        self.output_file = os.path.join(self.output_path, output_file)


# def main(
#     model_name: str,
#     run_name: str = None,
#     attn_type: str = "vllm",
#     output_path: str = "results/needle/",
#     pattern_path: str = "config/Llama_3_8B_Instruct_262k_kv_out_v32_best_pattern.json",
#     rounds: int = 3,
#     jobs: str = None,
#     max_length: int = 100000,
#     min_length: int = 1000,
#     kv_cache_cpu: bool = False,
#     trust_remote_code: bool = False,
#     kv_cache_cpu_device: str = "cpu",
#     kv_type: str = "dense",
# ):
def main(args):
    config = Config(
        model_name=args.model_name,
        output_path=args.output_path,
        n_rounds=args.rounds,
        jobs=args.jobs,
        context_lengths_min=args.min_length,
        context_lengths_max=args.max_length,
        n_context_length_intervals=args.n_context_length_intervals,
        n_document_depth_intervals=args.n_document_depth_intervals,
        trust_remote_code=args.trust_remote_code,
        args=args
    )

    try:
        ht = LLMNeedleHaystackTester(config,args=args)
        ht.start_test()

        print("making plot...")
        plot_needle_viz(
            config.output_file,
            (
                config.model_name.replace("/", "-") + f"_{config.run_name}"
                if config.run_name is not None
                else ""
            ),
            config.context_lengths_min,
            config.context_lengths_max,
            mode=config.real_model_name.replace("/","_"),
            output_path=config.output_path,
            show_here=False
        )
    except Exception as e:
        print(f"Error: {e}\n This occured during testing:{config.output_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    #NIAH
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--output_path", type=str, default="results/needle/")
    # args.add_argument("--pattern_path", type=str, default=None)
    args.add_argument("--rounds", type=int, default=3)
    args.add_argument("--jobs", type=str, default=None)
    args.add_argument("--max_length", type=int, default=100000)
    args.add_argument("--min_length", type=int, default=1000)
    args.add_argument("--trust_remote_code", action="store_true")
    args.add_argument("--device", type=str, default="auto")
    args.add_argument("--n_context_length_intervals", type=int, default=15)
    args.add_argument("--n_document_depth_intervals", type=int, default=10)
    
    #For all methods
    args.add_argument("--method", type=str, default="asl", choices=["vanilla","asl", "fastkv", "gemfilter", "pyramidinfer"])
    # Adaptive Select Layer
    def none_or_type(type_func):
        def convert(val):
            if val in ("None", "", None):
                return None
            return type_func(val)
        return convert
    args.add_argument("--default_select_layer", type=none_or_type(int), default=None)
    args.add_argument("--layer_th", type=none_or_type(float), default=None)
    args.add_argument("--select_k", type=none_or_type(int), default=None)
    args.add_argument("--window_size", type=none_or_type(int), default=None)
    args.add_argument("--kernel_size", type=none_or_type(int), default=None)
    args.add_argument("--use_layer_num", type=none_or_type(int), default=None)
    args.add_argument("--pooling", type=none_or_type(str), default=None)
    args.add_argument("--cache_k", type=none_or_type(int), default=None)
    args.add_argument("--cache_p", type=none_or_type(float), default=None)
    args = args.parse_args()

    
    main(args)
