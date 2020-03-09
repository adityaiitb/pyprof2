import torch
from apex.optimizers import FusedAdam
import amp_C
import pyprof2

pyprof2.init()
# Wrap the custom fused multi tensor Adam implementation
pyprof2.wrap(amp_C, 'multi_tensor_adam')

inp = 1024
hid = 2048
out = 4096
batch = 128

# Model
model = torch.nn.Sequential(
			torch.nn.Linear(inp, hid).cuda().half(),
			torch.nn.ReLU(),
			torch.nn.Linear(hid, out).cuda().half()
		)
# Loss
criterion = torch.nn.CrossEntropyLoss().cuda()
# Adam optimizer
optimizer = FusedAdam(model.parameters())
# Input
x = torch.ones(batch, inp).cuda().half()
# Target
target = torch.empty(batch, dtype=torch.long).random_(out).cuda()

with torch.autograd.profiler.emit_nvtx():
	y = model(x)
	loss = criterion(y, target)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
