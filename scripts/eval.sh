# DEVICES="0"
# NUM_WORKERS=0

# EVAL_DATASET="data/tsp/tsp10-200_concorde.txt"

# VAL_SIZE=25600

# MODEL2 = "pretrained/128K_20_wo_ss"
# MODELS=("outputs/tsp_20-50/rl-ar-var-20pnn-gnn-max_20200313T002243")



# BATCH_SIZE=8

# for MODEL in ${MODELS[*]}; do
#     echo $MODEL
#     CUDA_VISIBLE_DEVICES="$DEVICES" python eval.py  \
#         "$EVAL_DATASET" \
#         --val_size "$VAL_SIZE" --batch_size "$BATCH_SIZE" \
#         --model "$MODEL" \
#         --model2 "$MODEL2" \ 
#         --decode_strategies "greedy" \
#         --widths 128 \
#         --num_workers "$NUM_WORKERS" \
#         --cka
# done

# # For insertion baselines:
# # python eval_baseline.py random_insertion data/tsp/tsp10-200_concorde.txt -n 25600 --cpus 32 -f
#!/bin/bash

#!/bin/bash

DEVICES="0"
NUM_WORKERS=0

EVAL_DATASET="data_joshi/tsp/tsp20_test_concorde.txt"

VAL_SIZE=320

MODEL1="pretrained/384K_scaled_1ss_09"
MODEL2="pretrained/384k_scaled_no_ss"  # Replace this with the actual path to your second model

BATCH_SIZE=8

CUDA_VISIBLE_DEVICES="$DEVICES" python eval.py  \
    "$EVAL_DATASET" \
    --val_size "$VAL_SIZE" --batch_size "$BATCH_SIZE" \
    --model "$MODEL1" \
    --model2 "$MODEL2" \
    --decode_strategies "greedy" \
    --widths 128 \
    --num_workers "$NUM_WORKERS" \
    --cka

# For insertion baselines:
# python eval_baseline.py random_insertion data_joshi/tsp/tsp20_test_concorde.txt -n 128 --cpus 32 -f


# For insertion baselines:
# python eval_baseline.py random_insertion data_joshi/tsp/tsp20_test_concorde.txt -n 128 --cpus 32 -f
