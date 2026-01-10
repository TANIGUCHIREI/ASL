NIAH_MIN_LENGTH=4096
NIAH_MAX_LENGTH=262144

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

#    ./run_needle_core.sh $model None None None None $NIAH_MAX_LENGTH None vanilla $NIAH_MIN_LENGTH                                 # Vanilla
#    ./run_needle_core.sh $model None None $KV_Budget None $NIAH_MAX_LENGTH None vanill $NIAH_MIN_LENGTH                            # SnapKV
#    ./run_needle_core.sh $model $tsp_layer None None None $NIAH_MAX_LENGTH $select_k fastkv $NIAH_MIN_LENGTH                       # FastKV (FullKV before selection)
#    ./run_needle_core.sh $model $tsp_layer None $KV_Budget None $NIAH_MAX_LENGTH $select_k fastkv $NIAH_MIN_LENGTH                 # FastKV
   ./run_needle_core.sh $model $L_min $tau None None $NIAH_MAX_LENGTH $select_k asl $NIAH_MIN_LENGTH                              # ASL
#    ./run_needle_core.sh $model $L_min $tau $KV_Budget $cache_p $NIAH_MAX_LENGTH $select_k asl $NIAH_MIN_LENGTH                    # ASL (FullKV before selection)
#    ./run_needle_core.sh $model $filter_layer None None None $NIAH_MAX_LENGTH $gemfilter_select_k gemfilter $NIAH_MIN_LENGTH       # Gemfilter
#    ./run_needle_core.sh $model $L_min $tau None None $NIAH_MAX_LENGTH $gemfilter_select_k gemfilter $NIAH_MIN_LENGTH              # ASL_2pass
#    ./run_needle_core.sh $model None None None None $NIAH_MAX_LENGTH None pyramidinfer $NIAH_MIN_LENGTH                            # PyramidInfer