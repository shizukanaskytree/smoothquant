import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

# For TensorBoard visualization
from torch.utils.tensorboard import SummaryWriter

def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Save the model for visualization
    torch.save(model, "./logs/model.pt")
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--seq-len', type=int, default=512)
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = parse_args()
    print(f"model name: {args.model_name}")
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    # Set up TensorBoard
    writer = SummaryWriter('./logs/tensorboard_logs')

    # Tokenize a sample input
    sample_input = "This is a sample input for visualization."
    tokenized_input = tokenizer.encode(sample_input, return_tensors="pt", truncation=True, max_length=args.seq_len)

    # Add the model to TensorBoard with real input data
    writer.add_graph(model, tokenized_input)

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
