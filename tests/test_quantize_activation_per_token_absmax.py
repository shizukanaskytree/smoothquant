import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import sys
# Add the path of the folder containing the functions
sys.path.append('../smoothquant')

import torch

from smoothquant.fake_quant import quantize_activation_per_token_absmax

### example 1:
w = torch.tensor([
        [127.1, 254.1, 381.1],
        [254.9, 381.9, 508.9],
    ])

quantized_w = quantize_activation_per_token_absmax(w, n_bits=8)
print(quantized_w)

#-------------------------------------------------------------------------------

### example 2:
w = torch.tensor([
        [12.71, 25.41, 38.11],
        [25.49, 38.19, 50.89],
    ])
quantized_w = quantize_activation_per_token_absmax(w, n_bits=8)
print(quantized_w)
