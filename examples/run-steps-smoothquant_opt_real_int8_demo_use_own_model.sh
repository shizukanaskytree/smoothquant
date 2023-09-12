### goal is to run smoothquant_opt_real_int8_demo_use_own_model.ipynb
### - generate_act_scales.py
### - export_int8_model.py

#-----------------------------------------------------------------------

### execute
# setup_env.sh

#-----------------------------------------------------------------------

# python download_pile_val_dataset.py

git lfs install
git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup

#-----------------------------------------------------------------------

### activate conda env
# echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc # run at the first time
# source /opt/conda/etc/profile.d/conda.sh # run at the first time
conda activate smoothquant

#-----------------------------------------------------------------------
pip install transformers==4.33.1

### (smoothquant) root@b49e301cd39f:/workspace/outside-docker/smoothquant-prj/smoothquant/examples# pip show transformers
### Name: transformers
### Version: 4.33.1

#-----------------------------------------------------------------------

export MODEL_NAME='facebook/opt-125m'
export ACT_SCALES_PT_FILE='./act_scales/opt-125m.pt'
export DATASET_PATH='pile-val-backup/val.jsonl.zst'

mkdir -p logs

### for visualizing model
# pip install netron

python generate_act_scales.py \
    --model-name $MODEL_NAME \
    --output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH\
    --num-samples 512 \
    --seq-len 512
    # 2>&1 | tee logs/generate_act_scales_$(date +"%Y-%m-%d_%H-%M-%S").log

#-----------------------------------------------------------------------

python export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples 1000 \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --output-path 'int8_models'
    # 2>&1 | tee logs/export_int8_model_$(date +"%Y-%m-%d_%H-%M-%S").log

### go to smoothquant_opt_real_int8_demo_use_own_model.ipynb file and Run All
