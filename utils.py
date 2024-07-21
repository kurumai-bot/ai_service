from datetime import datetime, timedelta, timezone
import logging
from threading import Thread
import time
import traceback
from typing import Any, Dict, Tuple

import numpy as np

class CircularBuffer:
    def __init__(self, size: int, **kwargs) -> None:
        self.size = size
        self.dtype: np.dtype = kwargs.get("dtype", np.float32)
        self.index = 0
        self.buffer = np.empty(self.size, dtype=self.dtype)

    def add(self, data: np.ndarray) -> np.ndarray:
        # Number of return values will be current length of the buffer plus length of data
        # floor divided by buffer capacity
        ret = np.empty(((self.index + len(data)) // self.size, self.size))
        data_index = 0

        if len(ret) > 0:
            # Fill first ret value with a combination of current buffer data and passed in data
            ret[0][:self.index] = self.buffer[:self.index]
            data_index = self.size - self.index
            ret[0][self.index:] = data[:data_index]
            self.index = 0

            # Fill remaining buffers with however much data they can hold
            for i in range(1, len(ret)):
                ret[i] = data[data_index:data_index + self.size]
                data_index += self.size

        # Fill buffer with remaining data in passed in data
        start = self.index
        self.index = start + len(data) - data_index
        self.buffer[start:self.index] = data[data_index:]

        return ret

    def clear(self) -> None:
        self.index = 0

    def get(self) -> np.ndarray:
        return self.buffer[:self.index]

class Cache:
    def __init__(self, logger: logging.Logger = None) -> None:
        self.logger = logger or logging.getLogger("cache")
        self.items: Dict[str, Tuple[datetime, Any]] = {}

        self._thread = Thread(target=self._cache_loop)
        self._run_loop = True

    def __contains__(self, val: Any) -> bool:
        return val in self.items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, indices: Any) -> Tuple[datetime, Any] | None:
        return self.items[indices]

    def get(self, key: str) -> Tuple[datetime, Any] | None:
        return self.items.get(key)

    def add(self, key: str, item: Any, ttl: timedelta = None):
        if ttl is None:
            ttl = timedelta(minutes=15)

        expire_time = datetime.now(timezone.utc) + ttl
        self.items[key] = (expire_time, item)
        self.logger.debug("Item with key `%s` added, set to expire at %s.", key, expire_time)

    def remove(self, key: str) -> bool:
        if key not in self.items:
            return False

        item = self.items.pop(key)[1]

        # Release resources by calling __exit__
        exit_method = getattr(item, "__exit__", None)
        if callable(exit_method):
            exit_method(item)

        return True

    def start(self) -> None:
        self._run_loop = True
        self._thread.start()

    def __enter__(self) -> "Cache":
        self.start()
        return self

    def close(self) -> None:
        self._run_loop = False
        if self._thread.is_alive():
            self._thread.join()

        while len(self.items) > 0:
            self.remove(next(iter(self.items)))

    def __exit__(self, *args):
        self.close()

    def _cache_loop(self) -> None:
        while self._run_loop:
            try:
                if len(self.items) == 0:
                    time.sleep(1)
                    continue

                min_ = min(self.items.items(), key=lambda x: x[1][0])
                if min_[1][0] < datetime.now(timezone.utc):
                    self.remove(min_[0])
                    self.logger.debug("Removed expired item with key `%s`.", min_[0])

                time.sleep(1)
            except Exception: # pylint: disable=broad-exception-caught
                self.logger.error(
                    "Cache deletion loop failed with the following error:\n%s",
                    traceback.format_exc()
                )
        self.logger.info("Exited cache loop.")
