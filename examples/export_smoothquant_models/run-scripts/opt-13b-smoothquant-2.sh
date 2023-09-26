source opt-13b-smoothquant-1.sh

python ../generate_act_scales.py \
    --model-name $MODEL_NAME \
    --output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --num-samples $NUM_SAMPLES \
    --seq-len 512
