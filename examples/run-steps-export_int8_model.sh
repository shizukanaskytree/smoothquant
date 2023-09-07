### goal is to run smoothquant_opt_real_int8_demo_use_own_model.ipynb

### generate_act_scales.py
### export_int8_model.py

#-----------------------------------------------------------------------

### activate conda env
# echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc # run at the first time
# source /opt/conda/etc/profile.d/conda.sh # run at the first time
conda activate smoothquant

#-----------------------------------------------------------------------

# python download_pile_val_dataset.py

git lfs install
git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup

#-----------------------------------------------------------------------

MODEL_NAME='facebook/opt-125m'
ACT_SCALES_PT_FILE='./act_scales/opt-125m.pt'
DATASET_PATH='pile-val-backup/val.jsonl.zst'

python generate_act_scales.py \
    --model-name $MODEL_NAME \
    --output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH\
    --num-samples 512 \
    --seq-len 512

python export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples 1000 \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --output-path 'int8_models'

### go to smoothquant_opt_real_int8_demo_use_own_model.ipynb file and Run All

python debug_opt_real_int8.py


