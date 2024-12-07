# LICENSE
# This file was extracted from
#   https://github.com/LS4GAN/uvcgan2
# Please see `fccgan/base/LICENSE` for copyright attribution and LICENSE

import torch

class FastQueue:

    def __init__(self, size, device):
        self._queue    = None
        self._size     = size
        self._device   = device
        self._curr_idx = 0
        self._full     = False

    def __len__(self):
        if self._full:
            return self._size

        return self._curr_idx

    def lazy_init_queue(self, x):
        if self._queue is not None:
            return

        self._queue = torch.empty(
            (self._size, *x.shape[1:]), dtype = x.dtype, device = self._device
        )

    def push(self, x):
        self.lazy_init_queue(x)

        n = x.shape[0]
        n_avail_to_end = self._size - self._curr_idx

        if n > self._size:
            x = x[-self._size:, ...]
            n = self._size

        if n_avail_to_end <= n:
            self._queue[self._curr_idx:self._size, ...] \
                = x[:n_avail_to_end, ...].detach().to(
                    self._device, non_blocking = True
                )

            self._curr_idx = 0
            self._full     = True

            if n_avail_to_end < n:
                self.push(x[n_avail_to_end:, ...])

        else:
            self._queue[self._curr_idx:self._curr_idx + n, ...] \
                = x.detach().to(self._device)

            self._curr_idx += n

    def query(self):
        if self._queue is None:
            return None

        if self._full:
            return self._queue

        return self._queue[:self._curr_idx, ...]