# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")
TASKS=("kv_retrieval")
MAX_CONTEXT_LENGTH=131072

model="nvidia/Llama-3.1-8B-UltraLong-1M-Instruct" #["meta-llama/Llama-3.1-8B-Instruct","nvidia/Llama-3.1-8B-UltraLong-1M-Instruct","Qwen/Qwen2.5-7B-Instruct-1M","Qwen/Qwen2.5-14B-Instruct-1M"]

#FastKV
tsp_layer=15 
select_k=2048 #token selection length
KV_Budget=2048

#Gemfilter
filter_layer=13 
gemfilter_select_k=2048

#ASL
L_min=10
tau=0.3
select_k=2048 #token selection length
KV_Budget=2048



for task in ${TASKS[@]}; do 
#    ./run_infinitebench_per_task.sh $model None None None None $MAX_CONTEXT_LENGTH None vanilla $task                                 # Vanilla
#    ./run_infinitebench_per_task.sh $model None None $KV_Budget None $MAX_CONTEXT_LENGTH None vanilla $task                           # SnapKV
#    ./run_infinitebench_per_task.sh $model $tsp_layer None None None $MAX_CONTEXT_LENGTH $select_k fastkv $task                       # FastKV (FullKV before selection)
#    ./run_infinitebench_per_task.sh $model $tsp_layer None $KV_Budget None $MAX_CONTEXT_LENGTH $select_k fastkv $task                 # FastKV
   ./run_infinitebench_per_task.sh $model $L_min $tau None None $MAX_CONTEXT_LENGTH $select_k asl $task                              # ASL
#    ./run_infinitebench_per_task.sh $model $L_min $tau $KV_Budget $cache_p $MAX_CONTEXT_LENGTH $select_k asl $task                    # ASL (FullKV before selection)
   # ./run_infinitebench_per_task.sh $model $filter_layer None None None $MAX_CONTEXT_LENGTH $gemfilter_select_k gemfilter $task       # Gemfilter
   # ./run_infinitebench_per_task.sh $model $L_min $tau None None $MAX_CONTEXT_LENGTH $gemfilter_select_k gemfilter $task              # ASL_2pass
#    ./run_infinitebench_per_task.sh $model None None None None $MAX_CONTEXT_LENGTH None pyramidinfer $task                            # PyramidInfer
done