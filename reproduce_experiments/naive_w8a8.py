import argparse
from pathlib import Path

import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub import create_repo

# from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear

from evaluator import Evaluator

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    """ Code is adapted from the examples/smoothquant_opt_demo.ipynb

    The above function quantize_model is used to quantize specific layers of a
    given model. It checks for instances of OPTDecoderLayer and OPTAttention
    within the model and applies quantization to their respective attributes.
    """
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model

def upload_hf_hub(args, tokenizer, model_w8a8):
    """
    Save the mdoel and tokenizer locally and then upload to the remote HF hub.
    HF hub: https://huggingface.co/skytree/smoothquant-models/tree/main
    """
    local_saved_path = Path(args.smoothquant_int8_model_output) / args.model_name
    tokenizer.save_pretrained(local_saved_path)
    model_w8a8.save_pretrained(local_saved_path)

    ### tutorial: https://huggingface.co/docs/huggingface_hub/guides/repository
    try:
        create_repo(args.hf_repo_id)
    except:
        print("Repo already created.")

    api = HfApi()
    api.upload_folder(folder_path=local_saved_path, repo_id=args.hf_repo_id)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load a specific model.")

    parser.add_argument('--model_name', type=str, default='facebook/opt-125m',
        help='Name of the OPT model to load. facebook/opt-125m, facebook/opt-6.7b, facebook/opt-13b')

    parser.add_argument('--smoothquant-int8-model-output', type=str, default='',
        help='Saved local path')

    parser.add_argument('--hf-repo-id', type=str, default="skytree/smoothquant-models",
        help='HF Hub ID')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    model_name = args.model_name
    model_fp16 = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto')

    ### convert the model to W8A8
    model_w8a8 = quantize_model(model_fp16)
    # print(model_w8a8)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    if args.smoothquant_int8_model_output:
        upload_hf_hub(args, tokenizer, model_w8a8)
        print(f"Saved smoothed model at {args.smoothquant_int8_model_output}")

    dataset = load_dataset('lambada', split='validation[:1000]') # for testing
    # dataset = load_dataset('lambada', split='validation')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')

    acc_w8a8 = evaluator.evaluate(model_w8a8)
    print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')

if __name__ == '__main__':
    main()
