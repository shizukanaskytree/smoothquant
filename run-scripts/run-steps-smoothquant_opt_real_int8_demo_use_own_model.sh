### 1. Navigate to the directory where you want the symbolic link to be created. In your case, it's:
# cd /workspace/outside-docker/smoothquant-prj/smoothquant/examples

### 2. Create the symbolic link with the following command:
# ln -s /workspace/outside-docker/smoothquant-prj/smoothquant/run-scripts/run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh

#-----------------------------------------------------------------------

### goal is to run smoothquant_opt_real_int8_demo_use_own_model.ipynb
### - generate_act_scales.py
### - export_int8_model.py

#-----------------------------------------------------------------------

### execute
# setup_env.sh

#-----------------------------------------------------------------------

# python download_pile_val_dataset.py

# git lfs install
# git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup

#-----------------------------------------------------------------------

### activate conda env
# echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc # run at the first time
# source /opt/conda/etc/profile.d/conda.sh # run at the first time

# conda deactivate
# conda activate smoothquant

#-----------------------------------------------------------------------

### (smoothquant) root@b49e301cd39f:/workspace/outside-docker/smoothquant-prj/smoothquant/examples# pip show transformers
### Name: transformers
### Version: 4.33.1
# pip install transformers==4.33.1

#-----------------------------------------------------------------------

# MODEL_SIZE="opt-125m"
MODEL_SIZE="opt-6.7b"
# MODEL_SIZE="opt-13b"
MODEL_NAME="facebook/$MODEL_SIZE"

ACT_SCALES_PT_FILE="../examples/act_scales/$MODEL_SIZE.pt"
DATASET_PATH="../examples/pile-val-backup/val.jsonl.zst"
SMOOTHQUANT_OUTPUT="../examples/smoothquant-$MODEL_SIZE"

# mkdir -p logs

python ../examples/generate_act_scales.py \
    --model-name $MODEL_NAME \
    --scale-act-output-path $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --num-samples 512 \
    --seq-len 512
    # 2>&1 | tee logs/generate_act_scales_$(date +"%Y-%m-%d_%H-%M-%S").log

#-----------------------------------------------------------------------

python ../examples/export_int8_model.py \
    --model-name $MODEL_NAME \
    --num-samples 1000 \
    --seq-len 512 \
    --act-scales $ACT_SCALES_PT_FILE \
    --dataset-path $DATASET_PATH \
    --output-path 'int8_models' \
    --smoothquant-output $SMOOTHQUANT_OUTPUT
    # --torch-viewer \
    # 2>&1 | tee logs/export_int8_model_$(date +"%Y-%m-%d_%H-%M-%S").log

### view model
# dot -Tpdf ./logs/model.gv -o model-viewer/int8_model-depth-10.pdf

### go to smoothquant_opt_real_int8_demo_use_own_model.ipynb file and Run All
