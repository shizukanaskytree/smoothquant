# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

# %% [markdown]
# # SmoothQuant Real-INT8 Inference for PyTorch
#
# ### Guangxuan Xiao\*, Ji Lin\*, Mickael Seznec, Julien Demouth, Song Han
#
# In this notebook, we use OPT-30B model to demonstrate the latency and memory advantages of SmoothQuant. We implement SmoothQuant real-INT8 inference for PyTorch with [CUTLASS](https://github.com/NVIDIA/cutlass) INT8 GEMM kernels, which are wrapped as PyTorch modules in [torch-int](https://github.com/Guangxuan-Xiao/torch-int).
#
# This notebook demonstrates SmoothQuant on OPT-30B because it is the largest model we can run both FP16 and INT8 inference on a single A100 GPU. For larger models requiring multiple GPUs, we recommend using the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) implementation of SmoothQuant.
#
# In order to run this notebook, you need to install the following packages:
#
# - [smoothquant](https://github.com/mit-han-lab/smoothquant)
# - [torch-int](https://github.com/Guangxuan-Xiao/torch-int)
# - [PyTorch](https://pytorch.org/)
# - [Transformers](https://github.com/huggingface/transformers)
# - [Accelerate](https://github.com/huggingface/accelerate)

# %%
# model_type_size = "opt-125m"
# model_type_size = "opt-6.7b"
model_type_size = "opt-13b"

fb_model_name = f"facebook/{model_type_size}"

# %%
import torch
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.opt import Int8OPTForCausalLM
import os
import gc
from torch.nn.functional import pad

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# %% [markdown]
# The following is an evaluator to see the performance of the model. We use a toy dataset (the first 1000 examples in the validation set of the Lambada dataset) to evaluate the model. You can replace it with your dataset. The conclusion should be the same.

# %% [markdown]
# **In this demo, we have simplified the evaluation by using the first 1,000 samples from the LAMBADA dataset's validation set. We employ the "Last Token Prediction Accuracy" as our evaluation metric. This approximate evaluation is intended for demonstration purposes, providing simple but meaningful comparisons of relative performance between methods. For a more strict assessment, we recommend using the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) to obtain the "Last Word Prediction Accuracy" for the LAMBADA dataset, which is the reported metric in our paper.**

# %%
class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

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
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy


def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))


# %%
from datasets import load_dataset
tokenizer = GPT2Tokenizer.from_pretrained(fb_model_name)
dataset = load_dataset('lambada', split='validation[:1000]')
evaluator = Evaluator(dataset, tokenizer)

# %% [markdown]
# ## FP16 Model Accuracy and Latency

# %%
# model_fp16 = OPTForCausalLM.from_pretrained(
#     fb_model_name, torch_dtype=torch.float16, device_map='auto')
# print_model_size(model_fp16)
# acc_fp16, lantecy_fp16 = evaluator.evaluate(model_fp16)
# print(f'FP16 accuracy: {acc_fp16}, per-sample lantecy: {lantecy_fp16:.3f}ms')

# %%
# del model_fp16
# gc.collect()
# torch.cuda.empty_cache()

# %% [markdown]
# ## SmoothQuant W8A8 Quantized Model Accuracy and Latency

# %% [markdown]
# We provide the already smoothed and quantized OPT model at `https://huggingface.co/mit-han-lab/opt-[MODEL-SIZE]-smoothquant`, where `[MODEL-SIZE]` can be `125m`, `1.3B`, `2.7B`, `6.7B`, `13B`, `30b`, and `66b`. You can load the INT8 model with the following code:
#
# ```python
# from smoothquant.opt import Int8OPTForCausalLM
# model = Int8OPTForCausalLM.from_pretrained("mit-han-lab/opt-30b-smoothquant")
# ```
#
# We implement the following quantization flow for OPT models, which you can see details in [smoothquant/opt.py](../smoothquant/opt.py).
#
# ![quantization flow](../../figures/quantization_flow.png)
#
# You can also check [generate_act_scales.py](../examples/generate_act_scales.py) and [export_int8_model.py](../examples/export_int8_model.py) to see how we smooth, quantize and export INT8 models.

# %%
smoothquant_model_name = f"./smoothquant_output/{model_type_size}-smoothquant.pt"
print(smoothquant_model_name)

model_smoothquant = Int8OPTForCausalLM.from_pretrained(
    smoothquant_model_name, torch_dtype=torch.float16, device_map='cuda:0')

print_model_size(model_smoothquant)
acc_smoothquant, lantecy_smoothquant = evaluator.evaluate(model_smoothquant)
print(
    f'SmoothQuant INT8 accuracy: {acc_smoothquant}, per-sample lantecy: {lantecy_smoothquant:.3f}ms')


# %% [markdown]
# ## Conlusion
#
# We can see that the SmoothQuant model has a similar accuracy as the FP16 model, but it is faster and uses less memory. This is because SmoothQuant reduces the quantization difficulty of activations and enables the use of INT8 GEMM kernels.


