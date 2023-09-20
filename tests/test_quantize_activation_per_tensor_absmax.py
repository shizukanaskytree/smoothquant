# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import sys
# Add the path of the folder containing the functions
sys.path.append('../smoothquant')

import torch

from smoothquant.fake_quant import quantize_activation_per_tensor_absmax

### example 1:
w = torch.tensor([
        [127.0, 254.0],
        [254.0, 381.0],
    ])

quantized_w = quantize_activation_per_tensor_absmax(w, n_bits=8)
print(quantized_w)

#-------------------------------------------------------------------------------

### example 2:
w = torch.tensor([
        [12.70, 25.40],
        [25.40, 38.10],
    ])
quantized_w = quantize_activation_per_tensor_absmax(w, n_bits=8)
print(quantized_w)
