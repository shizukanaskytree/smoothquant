### 1. Navigate to the directory where you want the symbolic link to be created. In your case, it's:
# cd /workspace/outside-docker/smoothquant-prj/smoothquant/reproduce_experiments/

### 2. Create the symbolic link with the following command:
# ln -s /workspace/outside-docker/smoothquant-prj/smoothquant/run-scripts/run-naive-w8a8-benchmark.sh run-naive-w8a8-benchmark.sh

#-------------------------------------------------------------------------------

### setup env and conda to smoothquant
# bash setup_env.sh

#-------------------------------------------------------------------------------

# cd reproduce_experiments

#-------------------------------------------------------------------------------

# pip install --upgrade huggingface_hub
# apt-get install git-lfs
# git lfs install

#-------------------------------------------------------------------------------

MODEL_SIZE="opt-125m"
# MODEL_SIZE="opt-6.7b"
# MODEL_SIZE="opt-13b"
MODEL_NAME="facebook/$MODEL_SIZE"

LOCAL_SAVED_MODEL_PATH="./naive_w8a8_$MODEL_SIZE"
mkdir -p $LOCAL_SAVED_MODEL_PATH

FP16_MODEL_OUTPUT="./fp16_model_$MODEL_SIZE"

# HF_REPO_ID="skytree/naive-w8a8-$MODEL_SIZE"

### save the naive_w8a8 model
# start=`date +%s`
python export_naive_w8a8.py \
    --model_name $MODEL_NAME \
    --naive_w8a8_output $LOCAL_SAVED_MODEL_PATH \
    --fp16_model_output $FP16_MODEL_OUTPUT
# end=`date +%s`
# echo "Duartion: $((end-start)) seconds"

#-------------------------------------------------------------------------------

### Next step: eval the model using https://github.com/shizukanaskytree/lm-evaluation-harness/blob/dev-big-refactor-20230919/run-scripts/run-smoothquant.sh

### End