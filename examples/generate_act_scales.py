# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import torch
import os
from pathlib import Path
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from smoothquant.calibration import get_act_scales

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--scale-act-output-path', type=str, default='act_scales/opt-1.3b.pt',
                        help='where to save the act scales')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    print(f"model name: {args.model_name}")
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    ### debugging
    # for name, module in model.named_modules():
    #     if hasattr(module, 'weight') and module.weight is not None:
    #         # print(f"Module: {name} | Weight Shape: {module.weight.shape}")
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             print(f"Module: {name} | Bias Shape: {module.bias.shape}")

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    act_scales = get_act_scales(model, tokenizer, args.dataset_path,
                                args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(args.scale_act_output_path), exist_ok=True)
    torch.save(act_scales, args.scale_act_output_path)

if __name__ == '__main__':
    main()
