setup_env.sh

run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh

### create and get token
### https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=...

### https://huggingface.co/docs/huggingface_hub/guides/upload
huggingface-cli login --token $HUGGINGFACE_TOKEN

# In: (smoothquant) root@b0d7aa59d3f2:/workspace/outside-docker/smoothquant-prj/smoothquant/examples#
python upload_model_to_HF_hub.py
