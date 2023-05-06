import gc
import time
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class Timer:
    end_time: float = 0.0
    start_time: float = 0.0
    is_garbage_collector_disabled: bool = False

    def time_start(self, disable_garbage_collector: bool = True) -> Callable:
        def _time_start(
            module,
            args,
        ):
            if disable_garbage_collector:
                gc.disable()
                self.is_garbage_collector_disabled = True
            self.start_time = time.time()

        return _time_start

    def time_end(self, inference_times: List[float], name: str) -> Callable:
        def _time_end(module, args, output):
            self.end_time = time.time()

            # Automatically turn garbage collecting back on if it was turned off
            if self.is_garbage_collector_disabled:
                gc.enable()
                self.is_garbage_collector_disabled = False
            inference_times[name].append(self.end_time - self.start_time)

        return _time_end
