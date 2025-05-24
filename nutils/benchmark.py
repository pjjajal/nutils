import copy
import torch
import torch.nn as nn
from typing import Tuple, List
import torch.utils.benchmark as bench
from torchtnt.utils.flops import FlopTensorDispatchMode


__all__ = ["benchmark_model", "measure_flops"]


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int] | List[Tuple[int]],
    device: str,
    dtype: torch.dtype = torch.float32,
    min_run_time: float = 10.0,
    **kwargs,
):
    """
    Benchmarks the performance of a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to benchmark.
        input_shape (Tuple[int] | List[Tuple[int]]): The shape of the input tensor(s).
            Can be a single tuple or a list of tuples for multiple inputs.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        min_run_time (float, optional): The minimum run time for the benchmark in seconds.
            Defaults to 10.0 seconds.

    Returns:
        float: The average time per iteration in seconds.
    """
    # create inputs
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]
    inputs = [torch.randn(shape, dtype=dtype).to(device) for shape in input_shape]

    # move model to device
    model.to(device, dtype=dtype)
    model.eval()
    timer = bench.Timer(
        stmt="model.forward(*x, **kwargs)",
        globals={"model": model, "x": inputs, "kwargs": kwargs},
        num_threads=1,
    )
    return timer.blocked_autorange(min_run_time=min_run_time)


def measure_flops(
    model: nn.Module, input_shape: Tuple[int] | List[Tuple[int]], device: str, **kwargs
):
    """
    Measures the floating point operations (FLOPs) for the forward and backward passes of a given model.

    **NOTE: The measured FLOPs are a function of batch-size, to figure out FLOPs per sample, divide the total FLOPs by the batch-size.**

    Args:
        model (nn.Module): The neural network model to measure.
        input_shape (Tuple[int] | List[Tuple[int]]): The shape of the input tensor(s). Can be a single tuple or a list of tuples.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary with the following keys:
            - 'forward_per_module': A dictionary containing the FLOP counts for each module during the forward pass.
            - 'backward_per_module': A dictionary containing the FLOP counts for each module during the backward pass.
            - 'forward_total': The total FLOP count for the forward pass.
            - 'backward_total': The total FLOP count for the backward pass.
    """
    # create inputs
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]
    inputs = [torch.randn(shape).to(device) for shape in input_shape]
    
    # move model to device
    model = model.to(device)
    with FlopTensorDispatchMode(model) as ftdm:
        res = model(*inputs, **kwargs)
        flops_forward = copy.deepcopy(ftdm.flop_counts)
        ftdm.reset()
        if isinstance(res, torch.Tensor):
            res.sum().backward()
        if isinstance(res, (tuple, list)):
            res[0].sum().backward()
        flops_backward = copy.deepcopy(ftdm.flop_counts)

    total_forward = sum(
        [flops for val in flops_forward.values() for flops in val.values()]
    )
    total_backward = sum(
        [flops for val in flops_backward.values() for flops in val.values()]
    )
    return {
        "forward_per_module": flops_forward,
        "backward_per_module": flops_backward,
        "forward_total": total_forward,
        "backward_total": total_backward,
    }
