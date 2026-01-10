# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")
L_obs=8
kernel_size=7
window_size=32
pooling=avgpool
model_name=$1
default_select_layer=$2
layer_th=$3
cache_k=$4
cache_p=$5
max_seq_length=$6
select_k=$7
method=$8
task=$9
echo $task
python "$SCRIPT_DIR/run_infinitebench.py" \
    --task "$task" \
    --model_name "$model_name" \
    --pooling "$pooling" \
    --kernel_size "$kernel_size" \
    --window_size "$window_size" \
    --data_dir "$SCRIPT_DIR/data" \
    --output_dir "./outputs/infinite_bench" \
    --max_seq_length "$max_seq_length" \
    --default_select_layer "$default_select_layer" \
    --select_k "$select_k" \
    --use_layer_num "$L_obs" \
    --cache_k "$cache_k" \
    --cache_p "$cache_p" \
    --layer_th "$layer_th" \
    --method "$method" \
    --num_eval_examples -1
    # --rewrite \