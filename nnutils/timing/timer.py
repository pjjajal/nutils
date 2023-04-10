import time
from dataclasses import dataclass
from typing import List


@dataclass
class Timer:
    end_time: float = 0.0
    start_time: float = 0.0

    def time_start(self):
        def _time_start(
            module,
            args,
        ):
            self.start_time = time.time()

        return _time_start

    def time_end(self, inference_times: List[float]):
        def _time_end(module, args, output):
            self.end_time = time.time()
            inference_times.append(self.end_time - self.start_time)

        return _time_end
