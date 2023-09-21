# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import torch
import argparse
import os
from pathlib import Path

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi, create_repo

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm
from smoothquant.calibration import get_static_decoder_layer_scales

from torchview import draw_graph
# import graphviz
# graphviz.set_jupyter_format('png')

def model_viewer(model, tokenizer, filename: str):
    # print(f"int8_model: \n{int8_model}")

    device = next(model.parameters()).device
    ### Sample input for visualization, You can replace this with any sample text
    sample_text = "Hello, world!"

    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids.to(device)

    model_graph = draw_graph(
        model,
        input_data=input_ids,
        expand_nested=True,
        depth=10,
        hide_inner_tensors=False,
        hide_module_functions=False,
        directory='./logs',
        filename=filename,
        save_graph=True,
    )
    # dot -Tpdf ./logs/model.gv -o model-viewer/int8_model-depth-10.pdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='facebook/opt-13b')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/opt-13b.pt')
    parser.add_argument("--output-path", type=str, default='int8_models for the following ipynb demo example')
    parser.add_argument("--smoothquant-output", type=str, default=None,
                        help="where to save the original smoothquant int8 model")
    parser.add_argument("--hf_repo_id", type=str, default=None,
                        help="HF hub repo id, skytree/smoothquant-[MODEL_NAME], skytree/smoothquant-opt-125m, skytree/smoothquant-opt-6.7b, skytree/smoothquant-opt-13b")
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    parser.add_argument('--torch-viewer', default=False, action="store_true", help="visualize pytorch int8 models https://github.com/mert-kurttutan/torchview")
    args = parser.parse_args()

    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16)

    # print(f"OPTForCausalLM:\n{model}")

    ### debugging
    # for name, module in model.named_modules():
    #     if hasattr(module, 'weight') and module.weight is not None:
    #         # print(f"Module: {name} | Weight Shape: {module.weight.shape}")
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             print(f"Module: {name} | Bias Shape: {module.bias.shape}")

    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ### save tokenizer locally
    if args.smoothquant_output:
        tokenizer.save_pretrained(Path(args.smoothquant_output))

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    decoder_layer_scales, raw_scales = \
        get_static_decoder_layer_scales(
            model,
            tokenizer,
            args.dataset_path,
            num_samples=args.num_samples,
            seq_len=args.seq_len)

    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")

    if args.export_FT:
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        #-----------------------------------------------------------------------
        ### we are going to upload int8_model but we do not squeeze bias tensor etc.
        if args.smoothquant_output:
            smoothquant_model_origin = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
            smoothquant_model_origin.save_pretrained(args.smoothquant_output)
        #-----------------------------------------------------------------------

        ### fix bug to make model adapt to huggingface model shape for some tensors in the ipynb demo.
        for name, param in model.named_parameters():
            if 'bias' in name and 'final_layer_norm' not in name and 'self_attn_layer_norm' not in name:
                # Assuming you want to add a new dimension at the beginning of the tensor
                param.data = param.data.unsqueeze(0)

        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)

        ### save model for visualization
        if args.torch_viewer:
            # model_viewer(model, tokenizer, filename='model-OPTForCausalLM.gv')
            model_viewer(int8_model, tokenizer, filename='int8-model-Int8OPTForCausalLM.gv')

        ### debugging
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight') and module.weight is not None:
        #         # print(f"Module: {name} | Weight Shape: {module.weight.shape}")
        #         if hasattr(module, 'bias') and module.bias is not None:
        #             print(f"Module: {name} | Bias Shape: {module.bias.shape}")

        int8_model.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")
