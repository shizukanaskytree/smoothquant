### Code is adapted from the examples/smoothquant_opt_demo.ipynb

import torch

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
