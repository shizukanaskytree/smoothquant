#!/bin/bash

### setup env and conda to smoothquant
# bash setup_env.sh

#-------------------------------------------------------------------------------

# pip install --upgrade huggingface_hub
# apt-get install git-lfs
# git lfs install

#-------------------------------------------------------------------------------

MODEL_SIZES=(
    "opt-125m"
    "opt-6.7b"
    "opt-13b"
)

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    MODEL_NAME="facebook/$MODEL_SIZE"
    LOCAL_SAVED_MODEL_PATH="../naive_w8a8_${MODEL_SIZE}"
    mkdir -p $LOCAL_SAVED_MODEL_PATH

    ### save the naive_w8a8 model
    # start=`date +%s`
    python ../export_naive_w8a8.py \
        --model_name $MODEL_NAME \
        --naive_w8a8_output $LOCAL_SAVED_MODEL_PATH
    # end=`date +%s`
    # echo "Duartion: $((end-start)) seconds"
done

#-------------------------------------------------------------------------------

### Next step: eval the model using https://github.com/shizukanaskytree/lm-evaluation-harness/blob/dev-big-refactor-20230919
