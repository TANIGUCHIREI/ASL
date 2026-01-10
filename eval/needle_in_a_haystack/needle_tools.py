# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property

import numpy as np
import torch
from absl.app import run
from tqdm import tqdm
import logging

# from minference import MInference
import sys
sys.path.append("../../")


class LLMNeedleHaystackTester:
    OURS_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "
    RANDOM_NEEDLE_CITIES = [
        "Chicago",
        "Yangon",
        "Antananarivo",
        "Colombo",
        "Almaty",
        "Sydney",
        "Chicago",
        "Mexico City",
        "Seattle",
        "Lagos",
        "Amsterdam",
        "Belgrade",
        "Cairo",
        "Baghdad",
        "Damascus",
        "Kigali",
        "Dakar",
        "Dakar",
        "Sofia",
        "Kigali",
        "Victoria",
        "Tashkent",
        "Mumbai",
        "Barcelona",
        "Almaty",
        "Amman",
        "Toronto",
        "Bratislava",
        "Johannesburg",
        "Thimphu",
        "Bangkok",
        "Santiago",
        "Cairo",
        "San Francisco",
        "Lagos",
        "Amsterdam",
        "Paris",
        "Rabat",
        "Santiago",
        "Copenhagen",
        "Madrid",
        "Kigali",
        "Ho Chi Minh City",
        "Sarajevo",
        "Delhi",
        "Istanbul",
        "Ho Chi Minh City",
        "Khartoum",
        "Helsinki",
        "Doha",
        "Istanbul",
        "Kuala Lumpur",
        "Budapest",
        "Shanghai",
        "Moscow",
        "Los Angeles",
        "Oslo",
        "Johannesburg",
        "Berlin",
        "Bangalore",
        "Tokyo",
        "Melbourne",
        "Barcelona",
        "Chicago",
        "Port Louis",
        "Lisbon",
        "Nairobi",
        "Kampala",
        "Lima",
        "Maputo",
        "Vancouver",
        "Dubai",
        "Khartoum",
        "Jakarta",
        "Madrid",
        "Yerevan",
        "Beirut",
        "Athens",
        "Chicago",
        "Paris",
        "Bucharest",
        "Copenhagen",
        "Brussels",
        "Damascus",
        "Seattle",
        "Los Angeles",
        "Yerevan",
        "Victoria",
        "Tunis",
        "Astana",
        "Seoul",
        "Buenos Aires",
        "Bangkok",
        "Colombo",
        "Brussels",
        "Khartoum",
        "Doha",
        "San Francisco",
        "Vienna",
        "Jakarta",
    ]

    def __init__(
        self,
        config,
        retrieval_question= "What is the special magic {} number?",
        results_version=1,
        rnd_number_digits=7,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_interval_type="linear",
        save_results=False,
        final_context_length_buffer=200,
        print_ongoing_status=True,
        args=None,
    ):
        self.args = args
        
        haystack_file = config.haystack_file
        context_lengths_min = config.context_lengths_min
        context_lengths_max = config.context_lengths_max
        context_lengths_num_intervals = config.n_context_length_intervals
        document_depth_percent_intervals = config.n_document_depth_intervals

        self.config = config
        self.needle = "\nThe special magic {city} number is: {rnd_number}\n"
        # self.needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        # print(F"{haystack_file=},{retrieval_question=}")
        if not haystack_file or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.rnd_number_digits = rnd_number_digits
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.haystack_file = haystack_file
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.testing_results = []

        self.context_lengths = np.round(
            np.linspace(
                context_lengths_min,
                context_lengths_max,
                num=context_lengths_num_intervals,
                endpoint=True,
            )
        ).astype(int)
        if document_depth_percent_interval_type == "linear":
            self.document_depth_percents = np.round(
                np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    num=document_depth_percent_intervals,
                    endpoint=True,
                )
            ).astype(int)
        elif document_depth_percent_interval_type == "sigmoid":
            self.document_depth_percents = [
                self.logistic(x)
                for x in np.linspace(
                    document_depth_percent_min,
                    document_depth_percent_max,
                    document_depth_percent_intervals,
                )
            ]
        else:
            raise ValueError(
                f"Unsupported document_depth_percent_interval_type: {document_depth_percent_interval_type}"
            )

        if self.config.jobs is not None:
            start, end = self.config.jobs.split("-")
            print(self.context_lengths)
            self.context_lengths = self.context_lengths[int(start) : int(end)]
            print(self.context_lengths)

        # monkeypatch
        dtype=torch.bfloat16
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
            
        if args.model_name=="meta-llama/Meta-Llama-3.1-8B-Instruct" or args.model_name=="nvidia/Llama-3.1-8B-UltraLong-1M-Instruct":
            args.model_type="llama3.1"
        elif args.model_name=="Qwen/Qwen2.5-7B-Instruct-1M" or args.model_name=="Qwen/Qwen2.5-14B-Instruct-1M":
            args.model_type="qwen2.5"
        elif args.model_name=="microsoft/Phi-3.5-mini-instruct":
            args.model_type="phi3.5"
        else:
            ValueError(f"we dont support model: {args.model_name}")
            
        self.model =  model
        self.tokenizer = tokenizer

        self.generation_config = GenerationConfig(
            max_new_tokens=32,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
            


    def generate_random_number(self, num_digits):
        lower_bound = 10 ** (num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def read_context_files(self, n):
        max_context_length = max(self.context_lengths)
        contexts = []
        f = open(self.haystack_file, "r")
        for _ in range(n):
            context = ""
            toks = 0
            while toks < max_context_length:
                text = json.loads(f.readline())["text"]
                context += text
                toks += len(self.tokenizer.encode(text))
            contexts.append(context)
        return contexts

    def create_contexts(
        self,
        needle_rnd_number,
        insert_needle,
        random_city,
        trim_context,
        context_length,
        depth_percent,
        seed,
    ):
        needle = self.needle.format(city=random_city, rnd_number=needle_rnd_number)
        question = self.retrieval_question.format(random_city)
        if not insert_needle:
            needle = " "  # replace needle with a space
        context = self.insert_needle(
            needle, trim_context, depth_percent, context_length
        )
        results = {
            "context": context,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "needle": needle,
            "question": question,
            "insert_needle": insert_needle,
            "needle_rnd_number": needle_rnd_number,
            "seed": seed,
        }
        return results

    def insert_needle(self, needle, context, depth_percent, context_length):
        tokens_needle = self.tokenizer.encode(needle)
        tokens_context = self.tokenizer.encode(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.tokenizer.encode(".", add_special_tokens=False)

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.tokenizer.decode(tokens_new_context)
        return new_context

    def run_test(self):
        contexts = []
        template = self.OURS_TEMPLATE

        def _key_from_result(result):
            return (result["context_length"], result["depth_percent"], result["seed"])

        results = []
        full_contexts = self.read_context_files(self.config.n_rounds)
        full_tokens = [
            self.tokenizer.encode(full_context) for full_context in tqdm(full_contexts)
        ]

        start = time.time()
        for context_length in self.context_lengths:
            torch.cuda.empty_cache()
            trim_contexts = [
                self.tokenizer.decode(full_token[:context_length])
                for full_token in tqdm(full_tokens)
            ]
            contexts = []
            for depth_percent in self.document_depth_percents:
                for i in range(self.config.n_rounds):
                    random_city = random.choice(
                        LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES
                    )
                    insert_needle = True
                    needle_rnd_number = str(
                        self.generate_random_number(self.rnd_number_digits)
                    )
                    print("context length: " + str(context_length))
                    print("depth_percent : " + str(depth_percent))
                    context = self.create_contexts(
                        needle_rnd_number,
                        insert_needle,
                        random_city,
                        trim_contexts[i],
                        context_length,
                        depth_percent,
                        i,
                    )
                    contexts.append(context)

            for context in tqdm(contexts):
                prompt = template.format(
                    context=context["context"], question=context["question"]
                )

                msgs = [dict(role="system", content=prompt)]
                input_text = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False
                )
                input_tokens =self.tokenizer.encode(input_text)
                input_tensors = {
                    "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(self.model.device)
                }
                # set args
                self.model.cache_kwargs = self.args


                input_length = len(input_tokens)
                prefill_time,output_time,output_length, = None,None,None
                # outputs = model.generate(**input_tensors, generation_config=generation_config)
                if "gemfilter" in self.args.method:
                    from baseline.gemfilter.gemfilter_utils import gemfilter_generate_selection, set_topk
                    set_topk(self.model, self.args.select_k, mode='gemfilter')
                    output,prefill_time,output_time,output_length = gemfilter_generate_selection(
                        input_tensors["input_ids"], None, self.model, self.tokenizer, select_layer_idx=self.args.default_select_layer)
                else:
                    pred = self.model.generate(**input_tensors,generation_config=self.generation_config)
                    pred = pred[0, len(input_tokens) :]
                    output_length = pred.shape[0]
                    pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                    output = pred.strip()
                results.append(
                    {
                        "context_length": context["context_length"],
                        "depth_percent": context["depth_percent"],
                        "response": output,
                        "answer": context["needle_rnd_number"],
                        "correct": context["needle_rnd_number"] in output,
                        "seed": context["seed"],
                    }
                )
                
                analysis_data = {}
                analysis_data = {}
                if "gemfilter" in self.args.method:
                    analysis_data["prefill_time"] = prefill_time
                    analysis_data["output_time"] = output_time
                    analysis_data["output_time"] = output_time
                    analysis_data["input_length"] = input_length
                    analysis_data["output_length"] = output_length
                    allocated=0
                    for i in range(torch.cuda.device_count()):
                        allocated += torch.cuda.memory_allocated(i) 
                    analysis_data["allocated_memory"] = allocated
                    analysis_data["select_layer"] = self.model.gemfilter_select_layer
                else:
                    analysis_data =self.model.model.past_key_values.get_analysis_data()
                    analysis_data =self.model.model.past_key_values.get_analysis_data()
                    analysis_data["output_length"] = output_length
                    analysis_data["input_length"] = input_length

                results[-1] |=analysis_data
                
            with open(self.config.output_file, "w") as f:
                json.dump(results, f)
        print("elapsed", time.time() - start)
        print("done")
        print(f"Saved results to {self.config.output_file}")

    def print_start_test_summary(self):
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()