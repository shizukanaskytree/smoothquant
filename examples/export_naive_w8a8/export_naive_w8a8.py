import argparse
from pathlib import Path

import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from datasets import load_dataset

# from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear

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

class Evaluator:
    """
    In this demo, we have simplified the evaluation by using the first 1,000
    samples from the LAMBADA dataset's validation set. We employ the "Last Token
    Prediction Accuracy" as our evaluation metric. This approximate evaluation
    is intended for demonstration purposes, providing simple but meaningful
    comparisons of relative performance between methods. For a more strict
    assessment, we recommend using the
    [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) to
    obtain the "Last Word Prediction Accuracy" for the LAMBADA dataset, which is
    the reported metric in our paper.
    """
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

def parse_arguments():
    parser = argparse.ArgumentParser(description="Load a specific model.")
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m',
        help='Name of the OPT model to load. facebook/opt-125m, facebook/opt-6.7b, facebook/opt-13b')
    parser.add_argument('--naive_w8a8_output', type=str, default='',
        help='Saved w8a8 model to local path')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    model_name = args.model_name
    model_fp16 = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')

    ### convert the model to W8A8
    model_w8a8 = quantize_model(model_fp16)
    # print(model_w8a8)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    w8a8_saved_path = Path(args.naive_w8a8_output) / args.model_name
    tokenizer.save_pretrained(w8a8_saved_path)
    model_w8a8.save_pretrained(w8a8_saved_path)

    dataset = load_dataset('lambada', split='validation[:1000]') # for testing
    # dataset = load_dataset('lambada', split='validation')
    evaluator = Evaluator(dataset, tokenizer, 'cuda')

    acc_w8a8 = evaluator.evaluate(model_w8a8)
    print(f'Naive W8A8 quantized model accuracy: {acc_w8a8}')

if __name__ == '__main__':
    main()
