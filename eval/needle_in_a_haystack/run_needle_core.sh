# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false

# Load Haystack 
# mkdir -p data
# wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl


# Run the Needle in A Haystack Test

L_obs=8
kernel_size=7
window_size=32
pooling=avgpool
model_name=$1
default_select_layer=$2
layer_th=$3
cache_k=$4
cache_p=$5
max_length=$6
select_k=$7
method=$8
min_length=$9

n_context_length_intervals=10 #x axis count
n_document_depth_intervals=10 #y axis count
n_rounds=5

CUDA_VISIBLE_DEVICES=0
# ./run_needle_llama3_cuda1.sh nvidia/Llama-3.1-8B-UltraLong-1M-Instruct None None None None None vanilla　を実行
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ./needle_test.py \
    --model_name ${model_name} \
    --max_length $max_length \
    --min_length $min_length \
    --rounds $n_rounds \
    --output_path ./needle \
    --pooling avgpool \
    --kernel_size $kernel_size \
    --window_size $window_size \
    --default_select_layer $default_select_layer \
    --select_k $select_k \
    --use_layer_num $L_obs \
    --cache_k $cache_k \
    --cache_p $cache_p \
    --layer_th $layer_th \
    --method $method \
    --n_context_length_intervals $n_context_length_intervals \
    --n_document_depth_intervals $n_document_depth_intervals
