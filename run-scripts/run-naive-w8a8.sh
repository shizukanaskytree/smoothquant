### setup env and conda to smoothquant
bash setup_env.sh

#-------------------------------------------------------------------------------

cd reproduce_experiments

# MODEL_NAME="facebook/opt-125m"
MODEL_NAME="facebook/opt-6.7b"
# MODEL_NAME="facebook/opt-13b"

python naive_w8a8.py --model_name $MODEL_NAME

#-------------------------------------------------------------------------------
