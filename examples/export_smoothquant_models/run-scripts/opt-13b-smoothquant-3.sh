source opt-13b-smoothquant-1.sh

python ../export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples $NUM_SAMPLES \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --output-path $SMOOTHQUANT_OUTPUT \
    --dataset-path $DATASET_PATH
