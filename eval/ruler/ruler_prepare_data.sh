#!/bin/bash

# Root Directories

declare -A templete_dict
templete_dict["meta-llama/Meta-Llama-3.1-8B-Instruct"]="llama3.1"
templete_dict["nvidia/Llama-3.1-8B-UltraLong-1M-Instruct"]="llama3.1"
templete_dict["Qwen/Qwen2.5-7B-Instruct"]="qwen2.5"
templete_dict["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5"
templete_dict["microsoft/Phi-3.5-mini-instruct"]="phi3.5"

MODEL_PATH=nvidia/Llama-3.1-8B-UltraLong-1M-Instruct
# MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
# MODEL_PATH=microsoft/Phi-3.5-mini-instruct


max_seq_length_list=(4096 8192 16384 32768 65536 131072)
# MAX_SEQ_LENGTH=1024
BENCHMARK=synthetic
NUM_SAMPLES=200

# Benchmark and Tasks
source ruler_config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
TEMPLATE_TYPE=${templete_dict[$MODEL_PATH]}

for MAX_SEQ_LENGTH in ${max_seq_length_list[@]}; do
for TASK in ${synthetic[@]}; do

DATA_DIR="./created_data/${TEMPLATE_TYPE}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
mkdir -p ${DATA_DIR}

python -u data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${MODEL_PATH} \
    --tokenizer_type "hf" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type $TEMPLATE_TYPE \
    --num_samples ${NUM_SAMPLES}
done
done
