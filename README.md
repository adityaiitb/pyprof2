## PyProf2 - PyTorch Profiling tool

### What does this tool do?

Analyzing the performance of deep neural networks is hard. Getting kernels out of [NVProf]([https://developer.nvidia.com/nvidia-visual-profiler](https://developer.nvidia.com/nvidia-visual-profiler)) or [NSight Compute]([https://developer.nvidia.com/nsight-compute](https://developer.nvidia.com/nsight-compute)) provides some generic kernel names and execution times, but not detailed information regarding the following:

 - Which layer launched it: e.g. the association of `ComputeOffsetsKernel` with a concrete PyTorch layer or API is not obvious.
 - What the tensor dimensions and precision were: without knowing the tensor dimensions and precision, it's impossible to reason about whether the actual (silicon) kernel time is close to maximum performance of such a kernel on the GPU. Knowing the tensor dimensions and precision, we can figure out the FLOPs and bandwidth required by a layer, and then determine how close to maximum performance the kernel is for that operation.
 - Forward-backward correlation: currently it's very hard to determine what the forward pass step was that resulted in the particular weight and data gradients (wgrad, dgrad), which makes it difficult to determine the tensor dimensions required by these backprop steps to assess their performance.
 - Did the kernel use [Tensor Cores]([https://www.youtube.com/watch?v=yyR0ZoCeBO8](https://www.youtube.com/watch?v=yyR0ZoCeBO8))?
 - Which line in the user's code resulted in launching this particular kernel (program trace)?

PyProf addresses all of the issues above by:

 1. Instrumenting PyTorch operations to capture the tensor dimensions and precision using [NVTX](https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx). This information is recorded at profile capture time, e.g. using [NvProf](https://developer.nvidia.com/nvidia-visual-profiler).
 2. Querying the record produced by the profiler to correlate the kernel name and duration with PyTorch API/layer name, tensor dimensions, tensor precision, as well as calculating FLOPs and bandwidth for common operations. In addition, extra information from the profile is added for use by CUDA professionals, such as CUDA launch parameters (block/grid dimensions).

Regarding FLOP and bandwidth implementations, these are usually quite straightforward. For example, for matrices A<sub>MxK</sub> and B<sub>KxN</sub>, the FLOP count for a matrix multiplication is 2 * M * N * K, and bandwidth is M * K + N * K + M * N. Note that these numbers are based on the algorithm, not the actual performance of the specific kernel. For more details, see NVIDIA's [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).

Armed with such information, the user can determine various issues to help them tune the network. For instance, according to the [Tensor Core Performance Guide]([https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html)), the M, N and K dimensions that result in Tensor Core usage need to be divisible by 8. In fact, PyProf comes with a flag that lets the user obtain information regarding whether Tensor Cores were used by the kernel. Other useful information might include knowing that a particular kernel did not exploit much thread parallelism, as determined by the grid/block dimensions. Since many PyTorch kernels are open-source (or even custom written by the user, as in [CUDA Extensions]([https://pytorch.org/tutorials/advanced/cpp_extension.html](https://pytorch.org/tutorials/advanced/cpp_extension.html))), this provides the user with information that helps root cause performance issues and prioritize optimization work.

### To install / uninstall

```bash
pip3 install . --user
pip3 uninstall pyprof2
```

### How to get started?

1. Add the following lines to your PyTorch network:

    ```python
    import torch.cuda.profiler as profiler
    import pyprof2
    pyprof2.init()
    ```

    Run the training/inference loop with the [PyTorch's NVTX context manager](https://pytorch.org/docs/stable/_modules/torch/autograd/profiler.html#emit_nvtx)
    `with torch.autograd.profiler.emit_nvtx()`. In addition,
    use `profiler.start()` and `profiler.stop()` to pick an iteration(s)
    (say after warm-up) for which you would like to capture data.
    Here's an example:

    ```python
    iters = 500
    iter_to_capture = 100

    # Define network, loss function, optimizer etc.

    # PyTorch NVTX context manager
    with torch.autograd.profiler.emit_nvtx():

        for iter in range(iters):

            if iter == iter_to_capture:
                profiler.start()

            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if iter == iter_to_capture:
                profiler.stop()
    ```

2. Run NVprof to generate a SQL (NVVP) file. This file can be opened with NVVP, as usual.

    ```bash
    # If you used profiler.start() and profiler.stop()
    nvprof -f -o net.sql --profile-from-start off -- python net.py

    # If you did not use profiler.start() and profiler.stop() and want to profile everything
    nvprof -f -o net.sql -- python net.py
    ```

**Note:** if you're experiencing issues with hardware counters and you get a message such as `**_ERR_NVGPUCTRPERM The user running <tool_name/application_name> does not have permission to access NVIDIA GPU Performance Counters on the target device_**`, please follow the steps described in [Hardware Counters](#hardware-counters).

3. Run parser on the SQL file. The output is an ASCII file. Each line
is a python dictionary which contains information about the kernel name,
duration, parameters etc. This file can be used as input to other custom
scripts as well.

    ```bash
    pyprof2/parse/parse.py net.sql > net.dict
    ```

4. Run the profiler. The input is the python dictionary created above.

    ```bash
	# Columnated output of width 150 with some default columns.
    pyprof2/prof/prof.py -w 150 net.dict
    ```

The next section describes the various command line options for `prof.py`.

### Command line options for `prof.py`

`prof.py` can produce a CSV output, a columnated output (similar to
`column -t` for terminal readability) and a space separated output
(for post processing by AWK for instance).  The tool produces about 20
columns of information for every GPU kernel. You can select a subset
of columns and their order using the `-c` flag. Note that a few columns
might have the value *"na"* implying either its a work in progress or the
tool was unable to extract that information. Assuming the directory is
`pyprof2/prof`, here are a few examples of how to use `prof.py`.

```bash
# Print usage and lists all available output columns.
prof.py -h
```

Here is a list of columns from calling `prof.py -h`.

```
    idx:      Index
    seq:      PyTorch Sequence Id
    altseq:   PyTorch Alternate Sequence Id
    tid:      Thread Id
    layer:    User annotated NVTX string (can be nested)
    trace:    Function Call Trace
    dir:      Direction
    sub:      Sub Sequence Id
    mod:      Module
    op:       Operattion
    kernel:   Kernel Name
    params:   Parameters
    sil:      Silicon Time (in ns)
    tc:       Tensor Core Usage
    device:   GPU Device Id
    stream:   Stream Id
    grid:     Grid Dimensions
    block:    Block Dimensions
    flops:    Floating point ops (FMA = 2 FLOPs)
    bytes:    Number of bytes in and out of DRAM
```

```bash
# Columnated output of width 150 with some default columns.
prof.py -w 150 net.dict

# CSV output.
prof.py --csv net.dict

# Space seperated output.
prof.py net.dict

# Columnated output of width 130 with columns kernel,op,sil,tc,flops,bytes,device,stream,block,grid
prof.py -w 130 -c kernel,op,sil,tc,flops,bytes,device,stream,block,grid net.dict
```

```
Kernel              Op       Sil(ns) TC        FLOPs        Bytes Dev Str    Block          Grid
elementwise_kernel  relu      381028 -      51380224    205520896   0   7  512,1,1    100352,1,1 
volta_fp16_s884cudn conv2d    160002 1    1644167168     51388416   0   7  256,1,1       784,1,1 
elementwise_kernel  relu       96545 -      12845056     51380224   0   7  512,1,1     25088,1,1 
volta_fp16_s884cudn conv2d    346083 1    6576668672    128483328   0   7  256,1,1       784,2,1 
```

```bash
# CSV output with columns index,direction,kernel name,parameters,silicon time.
prof.py --csv -c idx,dir,kernel,params,sil net.dict

# Space separated output with columns index,direction,kernel name,parameters,silicon time.
prof.py -c idx,dir,kernel,params,sil net.dict

# Input redirection.
prof.py < net.dict
```

### Example

1. Run `nvprof` on the LeNet model in `examples/lenet.py`. This will
output a SQL file called `net.sql`.

    ```bash
    nvprof -f -o net.sql --profile-from-start off -- python examples/lenet.py
    ```

    (Optional) The SQL file can be opened with the NVIDIA Visual Profiler (NVVP)
    to view the timeline with detailed NVTX annotations inserted by PyProf2.

    ```bash
    nvvp net.sql
    ```

2. Run the `parse.py` script on `net.sql` to extract kernel and runtime
information and save it as `net.dict`.

    ```bash
    pyprof2/parse/parse.py net.sql > net.dict
    ```

This is an intermediate file. It is a Python dictionary per line. We
use it for debugging but we don't expect you to do anything with it.

3. Run `prof.py` on `net.dict` to get a CSV file.

    ```bash
    pyprof2/prof/prof.py --csv net.dict > net.csv
    ```

### Profile-guided optimization

If kernels that do matrix multiplication/GEMM or convolution use half
precision (fp16) data but do not use Tensor Cores (the TC column in the
profile analysis output doesn't show a "1"), one can follow some basic
steps to increase the likelihood that a Tensor Core-compatible kernel will
be chosen. For example, for GEMMs, M, N and K should be divisible by 8,
and for convolutions, the number of input and output channels shuold be
divisible by 8. For more information, see detailed Tensor Core guides
such as:
- Blog Post: [Tips for Optimizing GPU Performance Using Tensor Cores](https://devblogs.nvidia.com/optimizing-gpu-performance-tensor-cores/)
- GTC Talk: [Tensor Core Deep Learning Performance Guide](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf)

For both Tensor Core and non-Tensor Core Deep Learning performance
optimization tips, see NVIDIA's [Deep Learning Performance
Guide](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html).

### Hardware Counters

Profiling GPU workloads may require access to
[hardware performance counters]([https://en.wikipedia.org/wiki/Hardware_performance_counter](https://en.wikipedia.org/wiki/Hardware_performance_counter)).
Due to a
[fix](https://nvidia.custhelp.com/app/answers/detail/a_id/4738)
in recent NVIDIA drivers addressing
[CVE‑2018‑6260](https://nvd.nist.gov/vuln/detail/CVE-2018-6260),
the hardware counters are disabled by default, and require elevated
privileges to be enabled again. If you're using a recent driver, you
may see the following message when trying to run nvprof:

```**_ERR_NVGPUCTRPERM The user running <tool_name/application_name> does not have permission to access NVIDIA GPU Performance Counters on the target device._**```

For details, see [here](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters).

_Permanent solution_

Follow the steps [here]([https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)). The current steps for Linux are:
```
sudo systemctl isolate multi-user
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia-vgpu-vfio nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo systemctl isolate graphical
```
The above steps should result in a permanent change.

_Temporary solution_

When running on bare metal, you can run nvprof with `sudo`.

If you're running in a Docker image, you can temporarily elevate your privileges with one of the following (oldest to newest syntax):
<pre>
nvidia-docker run <b>--privileged</b>
docker run --runtime nvidia <b>--privileged</b>
docker run --gpus all <b>--privileged</b>
</pre>

### Notes

1. While running NVprof, do not add `--analysis-metrics`
since that will change which table NVprof writes the
kernels to (`CUPTI_ACTIVITY_KIND_KERNEL` instead of the usual
`CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`). Support for running with
metrics may be added in the future.

### TODOs
1. The support for conv transpose is currently missing.
2. PyProf currently works only with NvProf, but Nsight Compute support will be added in the future.
