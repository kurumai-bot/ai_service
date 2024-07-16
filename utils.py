from numpy import np

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
