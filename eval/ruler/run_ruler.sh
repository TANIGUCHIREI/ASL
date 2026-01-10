

TASKS=("niah_single_1" "niah_single_2" "niah_single_3" "niah_multikey_1" "niah_multikey_2" "niah_multikey_3" "niah_multivalue" "niah_multiquery" "vt" "cwe" "fwe" "qa_1" "qa_2")
TASKS=("niah_single_1")
MAX_CONTEXT_LENGTH=8192

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
#    ./run_ruler_per_task.sh $model None None None None $MAX_CONTEXT_LENGTH None vanilla $task                                 # Vanilla
#    ./run_ruler_per_task.sh $model None None $KV_Budget None $MAX_CONTEXT_LENGTH None vanilla $task                           # SnapKV
#    ./run_ruler_per_task.sh $model $tsp_layer None None None $MAX_CONTEXT_LENGTH $select_k fastkv $task                       # FastKV (FullKV before selection)
#    ./run_ruler_per_task.sh $model $tsp_layer None $KV_Budget None $MAX_CONTEXT_LENGTH $select_k fastkv $task                 # FastKV
   ./run_ruler_per_task.sh $model $L_min $tau None None $MAX_CONTEXT_LENGTH $select_k asl $task                              # ASL
#    ./run_ruler_per_task.sh $model $L_min $tau $KV_Budget $cache_p $MAX_CONTEXT_LENGTH $select_k asl $task                    # ASL (FullKV before selection)
   # ./run_ruler_per_task.sh $model $filter_layer None None None $MAX_CONTEXT_LENGTH $gemfilter_select_k gemfilter $task       # Gemfilter
#    ./run_ruler_per_task.sh $model $L_min $tau None None $MAX_CONTEXT_LENGTH $gemfilter_select_k gemfilter $task              # ASL_2pass
#    ./run_ruler_per_task.sh $model None None None None $MAX_CONTEXT_LENGTH None pyramidinfer $task                            # PyramidInfer
done