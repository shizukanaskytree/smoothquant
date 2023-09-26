export MODEL_SIZE="opt-13b"
export MODEL_NAME="facebook/$MODEL_SIZE"

export ACT_SCALES_PT_FILE="../../../act_scales/$MODEL_SIZE.pt"
export DATASET_PATH="../pile-val-backup/val.jsonl.zst"
export SMOOTHQUANT_OUTPUT="../smoothquant_output"

### if you use validation set, data leak?
export NUM_SAMPLES=512
