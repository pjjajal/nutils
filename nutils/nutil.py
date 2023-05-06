import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import objsize
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from .timing import Timer


@dataclass
class NUtil:
    modules_captured: Set[str] = field(default_factory=set)
    handles: Dict[str, List[RemovableHandle]] = field(default_factory=dict)
    data: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    chunk_num: int = 0

    def _check_module(self, name: str):
        """
        Checks if the data is being captured from an nn.Module labeled name.
        If it is not then the

        Args:
            name (str): Name of the module to add to data.
        """

        """
        Loops through all module names and guarantees that the names are present in the
        data sub-dictionaries .
        """
        for captured in self.modules_captured:
            self._init_name(captured)
        # Adds name if the name was not previously captured.
        if name not in self.modules_captured:
            self._init_name(name)

    def _data_exists(self, name: str):
        if name not in self.data:
            self.data[name] = {}

    def _init_name(self, name: str):
        # Add array to capture handles for that module.
        if name not in self.handles:
            self.handles[name] = []

        # Add name to the dictionaries in self.data if the name does not exist.
        for data_keys in self.data:
            if name not in self.data[data_keys]:
                self.data[data_keys][name] = []

        self.modules_captured.add(name)

    def time(
        self, module: nn.Module, name: str, disable_garbage_collector: bool = True
    ):
        self._data_exists("inference_time")
        self._check_module(name)

        timer = Timer()
        prehandle = module.register_forward_pre_hook(
            timer.time_start(disable_garbage_collector=disable_garbage_collector)
        )
        posthandle = module.register_forward_hook(
            timer.time_end(self.data["inference_time"], name)
        )
        self.handles[name].append(prehandle)
        self.handles[name].append(posthandle)

    def capture_activation(
        self, module: nn.Module, name: str, output_parser: Callable[..., Union[Tuple, Dict]]
    ):
        """
        This function is used to capture the activations (outputs) of a given nn.Module.
        An `output_parser` must be provided to parse the outputs of the module.

        Args:
            module (nn.Module): Module whose outputs will be captured.
            name (str): User-specified name given to `module`.
            output_parser (Callable[..., Tuple]): Parser used to parse the outputs of the module
            prior to saving. The output must be a `Tuple`.
        """
        self._data_exists("activations")
        self._check_module(name)

        def _capture_activation(module, inputs, output):
            self.data["activations"][name].append(output_parser(*output))

        handle = module.register_forward_hook(_capture_activation)
        self.handles[name].append(handle)

    def save(self, filename):
        torch.save(self.data, filename)

    def chunker_check(self, mem_limit: int = 1024**3):
        """Method used to chunk `data` if it takes up more space than `mem_limit`(default 1GB).

        Args:
            mem_limit (int, optional): The memory limit for the `data`. 
            If it becomes larger than `mem_limit` the chunker activates and persists the captured data
            and reduces memory size.
            Defaults to 1024**3.
        """
        mem_usage = objsize.get_deep_size(self.data)
        if mem_usage > mem_limit:
            print("Chunking!")
            self._chunk()

    def _chunk(self):
        self.chunk_num += 1
        torch.save(self.data, f"datachunk_{self.chunk_num}.nit")
        for subdict_name in self.data:
            for module in self.data[subdict_name]:
                self.data[subdict_name][module] = []
        gc.collect()
