import argparse
def none_or_type(type_func):
    def convert(val):
        if val in ("None", "", None):
            return None
        return type_func(val)
    return convert

def decide_name(args):
    model_name = args.model_name
    model_name = model_name.split("/")[-1]
    cache = "vanilla"
    if args.cache_k and args.cache_p:
        cache = f"pk-{args.cache_p}&&{args.cache_k}"
    elif args.cache_k:
        cache = f"k-{args.cache_k}"
    elif args.cache_p:
        cache = f"p-{args.cache_p}"

    if args.method=="asl":

        model_name = model_name + f"/{args.method}_l={args.default_select_layer}_k={args.select_k}_th={args.layer_th}_cache={cache}"
    elif args.method==args.method=="fastkv":
        model_name = model_name + f"/{args.method}_l={args.default_select_layer}_k={args.select_k}_cache={cache}"
    elif args.method =="gemfilter":
        model_name = model_name + f'/{args.method}_l={args.default_select_layer}_k={args.select_k}{f"_th={args.layer_th}" if args.layer_th else ""}'
    elif args.method=="pyramidinfer":
        model_name = model_name + f"/{args.method}"
    elif args.method=="vanilla":
        model_name = model_name + f'/{args.method}_cache={cache}{f"_th={args.layer_th}" if args.layer_th else ""}'
    else:
        ValueError(f"we do not support {args.method}")
    return model_name
if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="model name of model path")
    # Adaptive Select Layer
    parser.add_argument("--default_select_layer", type=none_or_type(int), default=None)
    parser.add_argument("--layer_th", type=none_or_type(float), default=None)
    parser.add_argument("--select_k", type=none_or_type(int), default=None)
    parser.add_argument("--window_size", type=none_or_type(int), default=None)
    parser.add_argument("--kernel_size", type=none_or_type(int), default=None)
    parser.add_argument("--use_layer_num", type=none_or_type(int), default=None)
    parser.add_argument("--pooling", type=none_or_type(str), default=None)
    parser.add_argument("--cache_k", type=none_or_type(int), default=None)
    parser.add_argument("--cache_p", type=none_or_type(float), default=None)
    parser.add_argument("--method", type=none_or_type(str), default=None)
    args = parser.parse_args()

    model_name = decide_name(args)
        
    print(model_name)