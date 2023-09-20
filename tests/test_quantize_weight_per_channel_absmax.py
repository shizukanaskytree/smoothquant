# the w is 2x2 tensor, the values are 1.1, 2.5, 0.5, -1.0 In this example, the
# input tensor w is a 2x2 tensor with values [1.1, 2.5] and [0.5, -1.0]. The
# quantize_weight_per_channel_absmax function is called with n_bits=8 to
# quantize the weights per channel using the absolute maximum value. The
# function returns the quantized weights, which are then printed as a tensor.

import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); debugpy.breakpoint()

import sys
# Add the path of the folder containing the functions
sys.path.append('../smoothquant')

import torch

from smoothquant.fake_quant import quantize_weight_per_channel_absmax

# Define the input tensor
w = torch.tensor([
        [1.1, 2.5],
        [0.5, -1.0]
    ])

# Call the quantize_weight_per_channel_absmax function
quantized_w = quantize_weight_per_channel_absmax(w, n_bits=8)

# Print the quantized weights
print(quantized_w)
