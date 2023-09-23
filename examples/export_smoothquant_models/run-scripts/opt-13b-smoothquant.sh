MODEL_SIZE="opt-13b"
MODEL_NAME="facebook/$MODEL_SIZE"

ACT_SCALES_PT_FILE="../act_scales/$MODEL_SIZE.pt"
DATASET_PATH="../examples/pile-val-backup/val.jsonl.zst"
SMOOTHQUANT_OUTPUT="../examples/smoothquant_output"

### if you use validation set, data leak?
NUM_SAMPLES=512

### 分开执行, 否则 OOM
# python ../examples/generate_act_scales.py \
#     --model-name $MODEL_NAME \
#     --output-path $ACT_SCALES_PT_FILE \
#     --dataset-path $DATASET_PATH \
#     --num-samples $NUM_SAMPLES \
#     --seq-len 512

# -----------------------------------------------------------------------

### 分开执行, 否则 OOM
python ../examples/export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples $NUM_SAMPLES \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --output-path $SMOOTHQUANT_OUTPUT \
    --dataset-path $DATASET_PATH
