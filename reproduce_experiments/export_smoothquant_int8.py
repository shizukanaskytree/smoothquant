### For OPT-13B, OOM, but if we run generate_act_scales.py and export_int8_model.py, it works.

### Commented for future debugging.


# # import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()
# import os
# from pathlib import Path
# import argparse

# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )
# from transformers.models.opt.modeling_opt import OPTForCausalLM

# from smoothquant.calibration import get_act_scales, get_static_decoder_layer_scales
# from smoothquant.opt import Int8OPTForCausalLM
# from smoothquant.smooth import smooth_lm

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model-name', type=str, default='facebook/opt-1.3b', help='model name')
#     parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst', help='location of the calibration dataset, we use the validation set of the Pile dataset')
#     parser.add_argument('--num-samples', type=int, default=512)
#     parser.add_argument('--seq-len', type=int, default=512)
#     parser.add_argument('--smoothquant-output', type=str, default='smoothquant-opt-6.7b', help="Where to save smoothquant int8 model.")
#     args = parser.parse_args()
#     return args

# def build_model_and_tokenizer(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
#     kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
#     model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
#     ### another way to build model specifically
#     # model = OPTForCausalLM.from_pretrained(
#     #    args.model_name, device_map="auto", torch_dtype=torch.float16)

#     return model, tokenizer

# def export_smoothquant_int8(args, model, tokenizer, act_scales):
#     smooth_lm(model, act_scales, 0.5)

#     if not os.path.exists(args.dataset_path):
#         print(f'Cannot find the dataset at {args.dataset_path}')
#         print('Please download the Pile dataset and put the validation set at the path')
#         print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
#         raise FileNotFoundError

#     decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(
#                                             model,
#                                             tokenizer,
#                                             args.dataset_path,
#                                             num_samples=args.num_samples,
#                                             seq_len=args.seq_len)

#     int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
#     int8_model.save_pretrained(args.smoothquant_output)
#     tokenizer.save_pretrained(args.smoothquant_output)

# @torch.no_grad()
# def main():
#     args = parse_args()
#     print(f"model name: {args.model_name}")

#     model, tokenizer = build_model_and_tokenizer(args.model_name)

#     if not os.path.exists(args.dataset_path):
#         print(f'Cannot find the dataset at {args.dataset_path}')
#         print('Please download the Pile dataset and put the validation set at the path')
#         print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
#         raise FileNotFoundError

#     act_scales = get_act_scales(model, tokenizer, args.dataset_path,
#                                 args.num_samples, args.seq_len)

#     ### smoothquant and then export processed model
#     export_smoothquant_int8(args, model, tokenizer, act_scales)

# if __name__ == '__main__':
#     main()
