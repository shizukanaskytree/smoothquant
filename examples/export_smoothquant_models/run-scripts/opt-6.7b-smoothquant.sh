MODEL_SIZE="opt-6.7b"
MODEL_NAME="facebook/$MODEL_SIZE"

ACT_SCALES_PT_FILE="../../../act_scales/$MODEL_SIZE.pt"
DATASET_PATH="../pile-val-backup/val.jsonl.zst"
SMOOTHQUANT_OUTPUT="../smoothquant_output"

### if you use validation set, data leak?
NUM_SAMPLES=512

python ../generate_act_scales.py \
    --model-name $MODEL_NAME \
    --scale-act-output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --num-samples $NUM_SAMPLES \
    --seq-len 512

# -----------------------------------------------------------------------

python ../export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples $NUM_SAMPLES \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --output-path $SMOOTHQUANT_OUTPUT \
    --dataset-path $DATASET_PATH
