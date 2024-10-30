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
    min_run_time: float = 10.0,
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
    inputs = [torch.randn(shape).to(device) for shape in input_shape]

    # move model to device
    model.to(device)
    model.eval()
    timer = bench.Timer(
        stmt="model.forward(*x)",
        globals={"model": model, "x": inputs},
        num_threads=1,
    )
    return timer.blocked_autorange(min_run_time=min_run_time)


def measure_flops(
    model: nn.Module, input_shape: Tuple[int] | List[Tuple[int]], device: str
):
    """
    Measures the floating point operations (FLOPs) for the forward and backward passes of a given model.

    Args:
        model (nn.Module): The neural network model to measure.
        input_shape (Tuple[int] | List[Tuple[int]]): The shape of the input tensor(s). Can be a single tuple or a list of tuples.
        device (str): The device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: A dictionary with two keys, 'forward' and 'backward', each containing the FLOP counts for the respective passes.
    """
    # create inputs
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]
    inputs = [torch.randn(shape).to(device) for shape in input_shape]

    # move model to device
    model.to(device)
    with FlopTensorDispatchMode(model) as ftdm:
        res = model(*inputs)
        flops_forward = copy.deepcopy(ftdm.flop_counts)
        ftdm.reset()
        res.sum().backward()
        flops_backward = copy.deepcopy(ftdm.flop_counts)
    return {"forward": flops_forward, "backward": flops_backward}
