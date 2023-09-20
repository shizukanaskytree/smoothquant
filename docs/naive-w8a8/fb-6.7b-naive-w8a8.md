Naive W8A8 quantized model

In this demo, we have simplified the evaluation by using the first 1,000 samples from the LAMBADA dataset's validation set. We employ the "Last Token Prediction Accuracy" as our evaluation metric. This approximate evaluation is intended for demonstration purposes, providing simple but meaningful comparisons of relative performance between methods. For a more strict assessment, we recommend using the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) to obtain the "Last Word Prediction Accuracy" for the LAMBADA dataset, which is the reported metric in our paper.

validation
- accuracy:

validation[:1000]
- accuracy: 0.396

model-00001-of-00002.bin: 9.96 GB
model-00002-of-00002.bin: 3.36 GB
