setup_env.sh

run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh

### create and get token
### https://huggingface.co/settings/tokens
HUGGINGFACE_TOKEN=...

### https://huggingface.co/docs/huggingface_hub/guides/upload
huggingface-cli login --token $HUGGINGFACE_TOKEN

### then you turn on True option on generate_act_scales.py and export_int8_model.py
### then run those two as run-steps-smoothquant_opt_real_int8_demo_use_own_model.sh