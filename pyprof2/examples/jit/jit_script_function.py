#!/usr/bin/env python3

import torch
import torch.cuda.profiler as profiler
import pyprof2

#The following creates an object "foo" of type ScriptModule
#The new object has a function called "forward"

@torch.jit.script
def foo(x, y):
	return torch.sigmoid(x) + y

#Initialize pyprof2 after the JIT step
pyprof2.init()

#Assign a name to the object "foo"
foo.__name__ = "foo"

#Hook up the forward function to pyprof2
pyprof2.wrap(foo, 'forward')

x = torch.zeros(4,4).cuda()
y = torch.ones(4,4).cuda()

with torch.autograd.profiler.emit_nvtx():
	profiler.start()
	z = foo(x, y)
	profiler.stop()
	print(z)
