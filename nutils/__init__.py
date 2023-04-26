import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .timing import Timer

__all__ = ["NUtil"]


@dataclass
class NUtil:
    modules_captured: List[str] = field(default_factory=list)
    handles: Dict[str, List[RemovableHandle]] = field(default_factory=dict)
    inference_times: Dict[str, List[float]] = field(default_factory=dict)
    activations: Dict[str, List[Tuple]] = field(default_factory=dict)

    def _check_module(self, name: str):
        if name not in self.modules_captured:
            self._init_name(name)

    def _init_name(self, name: str):
        if name not in self.handles:
            self.handles[name] = []

        if name not in self.inference_times:
            self.inference_times[name] = []

        if name not in self.activations:
            self.activations[name] = []

        self.modules_captured.append(name)

    def time(
        self, module: nn.Module, name: str, disable_garbage_collector: bool = True
    ):
        self._check_module(name)
        timer = Timer()
        prehandle = module.register_forward_pre_hook(
            timer.time_start(disable_garbage_collector=disable_garbage_collector)
        )
        posthandle = module.register_forward_hook(
            timer.time_end(self.inference_times[name])
        )
        self.handles[name].append(prehandle)
        self.handles[name].append(posthandle)

    def capture_activation(
        self, module: nn.Module, name: str, output_parser: Callable[..., Tuple]
    ):
        self._check_module(name)

        def _capture_activation(module, args, output):
            self.activations[name].append(output_parser(*output))

        handle = module.register_forward_hook(_capture_activation)
        self.handles[name] = handle
