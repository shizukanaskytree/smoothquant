### goal is to run smoothquant_opt_demo.ipynb

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

python generate_act_scales.py \
    --model-name 'facebook/opt-125m' \
    --output-path 'act_scales/opt-125m.pt' \
    --dataset-path 'pile-val-backup/val.jsonl.zst' \
    --num-samples 512 --seq-len 512

### go to smoothquant_opt_demo.ipynb file and Run All

#-----------------------------------------------------------------------

