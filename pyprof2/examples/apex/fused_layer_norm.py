import torch
import fused_layer_norm_cuda
from apex.normalization import FusedLayerNorm
import pyprof2

pyprof2.init()
pyprof2.wrap(fused_layer_norm_cuda, 'forward')
pyprof2.wrap(fused_layer_norm_cuda, 'backward')
pyprof2.wrap(fused_layer_norm_cuda, 'forward_affine')
pyprof2.wrap(fused_layer_norm_cuda, 'backward_affine')

input = torch.randn(20, 5, 10, 10).cuda()

# With Learnable Parameters
m = FusedLayerNorm(input.size()[1:]).cuda()
output = m(input)

# Without Learnable Parameters
m = FusedLayerNorm(input.size()[1:], elementwise_affine=False).cuda()
output = m(input)

# Normalize over last two dimensions
m = FusedLayerNorm([10, 10]).cuda()
output = m(input)

# Normalize over last dimension of size 10
m = FusedLayerNorm(10).cuda()
output = m(input)
