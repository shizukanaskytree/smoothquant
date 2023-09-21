### Go to folder examples/, run separately: run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh
### since OOM error will occur if run together.

################################################################################

### 1. Navigate to the directory where you want the symbolic link to be created. In your case, it's:
# cd /workspace/outside-docker/smoothquant-prj/smoothquant/reproduce_experiments/

### 2. Create the symbolic link with the following command:
# ln -s /workspace/outside-docker/smoothquant-prj/smoothquant/run-scripts/run-export-smoothquant-int8-model.sh run-export-smoothquant-int8-model.sh

#-----------------------------------------------------------------------

# python download_pile_val_dataset.py

#-----------------------------------------------------------------------

# git lfs install
# git clone https://huggingface.co/datasets/mit-han-lab/pile-val-backup

#-----------------------------------------------------------------------

### activate conda env
# echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc # run at the first time
# source /opt/conda/etc/profile.d/conda.sh # run at the first time
# conda activate smoothquant

#-----------------------------------------------------------------------

### (smoothquant) root@b49e301cd39f:/workspace/outside-docker/smoothquant-prj/smoothquant/examples# pip show transformers
### Name: transformers
### Version: 4.33.1
# pip install transformers==4.33.1

#-----------------------------------------------------------------------

# MODEL_SIZE="opt-125m"
# MODEL_SIZE="opt-6.7b"
# MODEL_SIZE="opt-13b"
# MODEL_NAME=facebook/$MODEL_SIZE

# # ln -s /workspace/outside-docker/smoothquant-prj/smoothquant/examples/pile-val-backup pile-val-backup
# DATASET_PATH='pile-val-backup/val.jsonl.zst'

# LOCAL_SAVED_MODEL_PATH=./smoothquant-$MODEL_SIZE
# mkdir -p $LOCAL_SAVED_MODEL_PATH

# ### note: num-samples 1000 will change mean
# python export_smoothquant_int8.py \
#     --model-name $MODEL_NAME \
#     --dataset-path $DATASET_PATH \
#     --seq-len 512 \
#     --num-samples 500 \
#     --smoothquant-output $LOCAL_SAVED_MODEL_PATH

#-----------------------------------------------------------------------

### Next step: eval the model using https://github.com/shizukanaskytree/lm-evaluation-harness/blob/dev-big-refactor-20230919/run-scripts/run-smoothquant.sh
