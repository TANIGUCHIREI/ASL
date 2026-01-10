#!/bin/bash
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


# ./scripts/run_infinitebench.sh Qwen/Qwen2.5-7B-Instruct-1M None None 2048 None 128000 None& infinite-benchと同じくこんな感じで実行したい

declare -A templete_dict
templete_dict["meta-llama/Meta-Llama-3.1-8B-Instruct"]="llama3.1"
templete_dict["nvidia/Llama-3.1-8B-UltraLong-1M-Instruct"]="llama3.1"
templete_dict["Qwen/Qwen2.5-7B-Instruct"]="qwen2.5"
templete_dict["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5"
templete_dict["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen2.5"
templete_dict["Qwen/Qwen2.5-14B-Instruct-1M"]="qwen2.5"
templete_dict["microsoft/Phi-3.5-mini-instruct"]="phi3.5"


ROOT_DIR="./ruler_eval_result" # the path that stores generated task samples and model predictions.
MODEL_PATH=$1 #[Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B-Instruct, meta-llama/Llama-3.1-8B-Instruct, microsoft/Phi-3.5-mini-instruct]
DTYPE=bf16
default_select_layer=$2
layer_th=$3
cache_k=$4
cache_p=$5
MAX_SEQ_LENGTH=$6
select_k=$7
method=$8
BENCHMARK=synthetic
NUM_SAMPLES=200
DEVICE=auto
use_layer_num=8
kernel_size=7
window_size=32
pooling=avgpool

TASK=$9
# Benchmark and Tasks
source ruler_config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
SAVE_MODEL_NAME="$(python -u ./decide_results_dir.py \
    --model_name "${MODEL_PATH}" \
    --pooling $pooling \
    --kernel_size "$kernel_size" \
    --window_size "$window_size" \
    --default_select_layer "$default_select_layer" \
    --select_k "$select_k" \
    --use_layer_num "$use_layer_num" \
    --cache_k "$cache_k" \
    --cache_p "$cache_p" \
    --layer_th "$layer_th" \
    --method "$method"
)"

TEMPLATE_TYPE=${templete_dict[$MODEL_PATH]}
RESULTS_DIR="${ROOT_DIR}/${SAVE_MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
DATA_DIR="./created_data/${TEMPLATE_TYPE}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
PRED_DIR="${RESULTS_DIR}"
LOG_DIR="${PRED_DIR}/logs"

mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}
mkdir -p ${LOG_DIR}


echo "start: ${TASK}"
python -u pred/call_api.py \
    --model_name ${MODEL_PATH} \
    --max_len ${MAX_SEQ_LENGTH} \
    --data_dir ${DATA_DIR} \
    --save_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --dtype ${DTYPE} \
    --server_type "hf" \
    --device ${DEVICE} \
    --synthetic_len ${MAX_SEQ_LENGTH} \
    --pooling avgpool \
    --kernel_size $kernel_size \
    --window_size $window_size \
    --default_select_layer $default_select_layer \
    --select_k $select_k \
    --use_layer_num $use_layer_num \
    --cache_k $cache_k \
    --cache_p $cache_p \
    --layer_th $layer_th \
    --method $method \
    > $LOG_DIR/${TASK}.log 2>&1 || exit 1  # stop this loop if failure/kill happen
python -u eval/evaluate.py \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}


