### setup env and conda to smoothquant
bash setup_env.sh

#-------------------------------------------------------------------------------

cd reproduce_experiments

MODEL_SIZE="opt-125m"
# MODEL_SIZE="opt-6.7b"
# MODEL_SIZE="opt-13b"

MODEL_NAME="facebook/$MODEL_SIZE"
LOCAL_SAVED_MODEL_PATH="/tmp"
HF_REPO_ID="skytree/naive-w8a8-$MODEL_SIZE"

start=`date +%s`
### save the naive_w8a8 model
python naive_w8a8.py \
    --model_name $MODEL_NAME \
    --smoothquant-int8-model-output $LOCAL_SAVED_MODEL_PATH \
    --hf-repo-id $HF_REPO_ID
end=`date +%s`

echo "Duration: $((($(date +%s)-$start)/60)) minutes

#-------------------------------------------------------------------------------

### eval the model using https://github.com/shizukanaskytree/lm-evaluation-harness/blob/dev-big-refactor-20230919/run-scripts/run-smoothquant.sh
