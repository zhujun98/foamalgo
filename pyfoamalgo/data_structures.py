"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
from abc import abstractmethod
from collections import deque, namedtuple, OrderedDict
from collections.abc import MutableSet, Sequence
from queue import Empty, Full
from threading import Lock

import numpy as np

from .lib.imageproc import movingAvgImageData


__all__ = [
    'OrderedSet',
    'Stack',
    'SimpleSequence',
    'SimpleVectorSequence',
    'SimplePairSequence',
    'OneWayAccuPairSequence',
    'MovingAverageScalar',
    'MovingAverageArray',
    'SimpleQueue',
]


class Stack:
    """An LIFO stack."""
    def __init__(self):
        self.__items = []

    def push(self, item):
        """Append a new element."""
        self.__items.append(item)

    def pop(self):
        """Return and remove the top element."""
        return self.__items.pop()

    def top(self):
        """Return the first element."""
        if self.empty():
            raise IndexError("Stack is empty")

        return self.__items[-1]

    def empty(self):
        return not self.__items

    def __len__(self):
        return len(self.__items)


class OrderedSet(MutableSet):
    def __init__(self, sequence=None):
        super().__init__()

        if sequence is None:
            self._data = OrderedDict()
        else:
            kwargs = {v: 1 for v in sequence}
            self._data = OrderedDict(**kwargs)

    def __contains__(self, item):
        """Override."""
        return self._data.__contains__(item)

    def __iter__(self):
        """Override."""
        return self._data.__iter__()

    def __len__(self):
        """Override."""
        return self._data.__len__()

    def add(self, item):
        """Override."""
        self._data.__setitem__(item, 1)

    def discard(self, item):
        """Override."""
        if item in self._data:
            self._data.__delitem__(item)

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._data.keys())})"


class _AbstractSequence(Sequence):
    """Abstract class for 'Sequence' data.

    It cannot be instantiated without subclassing.
    """
    _OVER_CAPACITY = 2

    def __init__(self, max_len=3000):
        self._max_len = max_len

        self._i0 = 0  # starting index
        self._len = 0

    def __len__(self):
        """Override."""
        return self._len

    @abstractmethod
    def data(self):
        """Return all the data."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the data history."""
        pass

    @abstractmethod
    def append(self, item):
        """Add a new data point."""
        pass

    @abstractmethod
    def extend(self, items):
        """Add a list of data points."""
        pass

    @classmethod
    def from_array(cls, *args, **kwargs):
        """Construct from array(s)."""
        raise NotImplementedError


class SimpleSequence(_AbstractSequence):
    """Store the history of scalar data."""

    def __init__(self, *, max_len=100000, dtype=np.float64):
        super().__init__(max_len=max_len)

        self._x = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)

    def __getitem__(self, index):
        """Override."""
        return self._x[self._i0:self._i0 + self._len][index]

    def data(self):
        """Override."""
        return self._x[slice(self._i0, self._i0 + self._len)]

    def append(self, item):
        """Override."""
        self._x[self._i0 + self._len] = item
        max_len = self._max_len
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len] = self._x[max_len:]

    def extend(self, items):
        for item in items:
            self.append(item)

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)

    @classmethod
    def from_array(cls, ax, *args, **kwargs):
        instance = cls(*args, **kwargs)
        for x in ax:
            instance.append(x)
        return instance


class SimpleVectorSequence(_AbstractSequence):
    """Store the history of vector data."""

    def __init__(self, size, *, max_len=100000, dtype=np.float64, order='C'):
        super().__init__(max_len=max_len)

        self._x = np.zeros((self._OVER_CAPACITY * max_len, size),
                           dtype=dtype, order=order)
        self._size = size

    @property
    def size(self):
        return self._size

    def __getitem__(self, index):
        """Override."""
        return self._x[self._i0:self._i0 + self._len, :][index]

    def data(self):
        """Override."""
        return self._x[slice(self._i0, self._i0 + self._len), :]

    def append(self, item):
        """Override.

        :raises: ValueError, if item has different size;
                 TypeError, if item has no method __len__.
        """
        if len(item) != self._size:
            raise ValueError(f"Item size {len(item)} differs from the vector "
                             f"size {self._size}!")

        self._x[self._i0 + self._len, :] = np.array(item)
        max_len = self._max_len
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len, :] = self._x[max_len:, :]

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)

    @classmethod
    def from_array(cls, ax, *args, **kwargs):
        instance = cls(*args, **kwargs)
        for x in ax:
            instance.append(x)
        return instance


class SimplePairSequence(_AbstractSequence):
    """Store the history a pair of scalar data.

    Each data point is pair of data: (x, y).
    """

    def __init__(self, *, max_len=3000, dtype=np.float64):
        super().__init__(max_len=max_len)
        self._x = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)

    def __getitem__(self, index):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)
        return self._x[s][index], self._y[s][index]

    def data(self):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)
        return self._x[s], self._y[s]

    def append(self, item):
        """Override."""
        x, y = item

        max_len = self._max_len
        self._x[self._i0 + self._len] = x
        self._y[self._i0 + self._len] = y
        if self._len < max_len:
            self._len += 1
        else:
            self._i0 += 1
            if self._i0 == max_len:
                self._i0 = 0
                self._x[:max_len] = self._x[max_len:]
                self._y[:max_len] = self._y[max_len:]

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Override."""
        self._i0 = 0
        self._len = 0
        self._x.fill(0)
        self._y.fill(0)

    @classmethod
    def from_array(cls, ax, ay, *args, **kwargs):
        if len(ax) != len(ay):
            raise ValueError(f"ax and ay must have the same length. "
                             f"Actual: {len(ax)}, {len(ay)}")

        instance = cls(*args, **kwargs)
        for x, y in zip(ax, ay):
            instance.append((x, y))
        return instance


_StatDataItem = namedtuple('_StatDataItem', ['avg', 'min', 'max', 'count'])


class OneWayAccuPairSequence(_AbstractSequence):
    """Store the history a pair of accumulative scalar data.

    Each data point is pair of data: (x, _StatDataItem).

    The data is collected in a stop-and-collected way. A motor, for
    example, will stop in a location and collect data for a period
    of time. Then, each data point in the accumulated pair data is
    the average of the data during this period.
    """

    def __init__(self, resolution, *,
                 max_len=3000, dtype=np.float64, min_count=2):
        super().__init__(max_len=max_len)

        self._min_count = min_count

        if resolution <= 0:
            raise ValueError("resolution must be positive!")
        self._resolution = resolution

        self._x_avg = np.zeros(self._OVER_CAPACITY * max_len, dtype=dtype)
        self._count = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=np.uint64)
        self._y_avg = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_min = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_max = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)
        self._y_std = np.zeros(
            self._OVER_CAPACITY * max_len, dtype=dtype)

        self._last = 0

    def __getitem__(self, index):
        """Override."""
        s = slice(self._i0, self._i0 + self._len)

        x = self._x_avg[s][index]
        y = _StatDataItem(self._y_avg[s][index],
                          self._y_min[s][index],
                          self._y_max[s][index],
                          self._count[s][index])
        return x, y

    def data(self):
        """Override."""
        last = self._i0 + self._len - 1
        if self._len > 0 and self._count[last] < self._min_count:
            s = slice(self._i0, last)
        else:
            s = slice(self._i0, last + 1)

        x = self._x_avg[s]
        y = _StatDataItem(self._y_avg[s],
                          self._y_min[s],
                          self._y_max[s],
                          self._count[s])
        return x, y

    def append(self, item):
        """Override."""
        x, y = item

        new_pt = False
        last = self._last
        if self._len > 0 or self._count[0] > 0:
            if abs(x - self._x_avg[last]) <= self._resolution:
                self._count[last] += 1
                self._x_avg[last] += (x - self._x_avg[last]) / self._count[last]
                avg_prev = self._y_avg[last]
                self._y_avg[last] += (y - self._y_avg[last]) / self._count[last]
                self._y_std[last] += (y - avg_prev)*(y - self._y_avg[last])
                # self._y_min and self._y_max does not store min and max
                # Only Standard deviation will be plotted. Min Max functionality
                # does not exist as of now.
                # self._y_min stores y_avg - 0.5*std_dev
                # self._y_max stores y_avg + 0.5*std_dev
                self._y_min[last] = self._y_avg[last] - 0.5*np.sqrt(
                    self._y_std[last]/self._count[last])
                self._y_max[last] = self._y_avg[last] + 0.5*np.sqrt(
                    self._y_std[last]/self._count[last])

                if self._count[last] == self._min_count:
                    new_pt = True

            else:
                # If the number of data at a location is less than
                # min_count, the data at this location will be discarded.
                if self._count[last] >= self._min_count:
                    self._last += 1
                    last = self._last

                self._x_avg[last] = x
                self._count[last] = 1
                self._y_avg[last] = y
                self._y_min[last] = y
                self._y_max[last] = y
                self._y_std[last] = 0.0

        else:
            self._x_avg[0] = x
            self._count[0] = 1
            self._y_avg[0] = y
            self._y_min[0] = y
            self._y_max[0] = y
            self._y_std[0] = 0.0

        if new_pt:
            max_len = self._max_len
            if self._len < max_len:
                self._len += 1
            else:
                self._i0 += 1
                if self._i0 == max_len:
                    self._i0 = 0
                    self._last -= max_len
                    self._x_avg[:max_len] = self._x_avg[max_len:]
                    self._count[:max_len] = self._count[max_len:]
                    self._y_avg[:max_len] = self._y_avg[max_len:]
                    self._y_min[:max_len] = self._y_min[max_len:]
                    self._y_max[:max_len] = self._y_max[max_len:]
                    self._y_std[:max_len] = self._y_std[max_len:]

    def append_dry(self, x):
        """Return whether append the given item will start a new position."""
        next_pos = False
        if self._len > 0 or self._count[0] > 0:
            if abs(x - self._x_avg[self._last]) > self._resolution:
                next_pos = True
        else:
            next_pos = True

        return next_pos

    def extend(self, items):
        """Override."""
        for item in items:
            self.append(item)

    def reset(self):
        """Overload."""
        self._i0 = 0
        self._len = 0
        self._last = 0
        self._x_avg.fill(0)
        self._count.fill(0)
        self._y_avg.fill(0)
        self._y_min.fill(0)
        self._y_max.fill(0)
        self._y_std.fill(0)

    @classmethod
    def from_array(cls, ax, ay, *args, **kwargs):
        if len(ax) != len(ay):
            raise ValueError(f"ax and ay must have the same length. "
                             f"Actual: {len(ax)}, {len(ay)}")

        instance = cls(*args, **kwargs)
        for x, y in zip(ax, ay):
            instance.append((x, y))
        return instance


class _MovingAverageBase:
    def __init__(self, window=1):
        """Initialization.

        :param int window: moving average window size.
        """
        self._data = None  # moving average

        if not isinstance(window, int) or window < 0:
            raise ValueError("Window must be a positive integer.")

        self._window = window
        self._count = 0

    def __get__(self, instance, instance_type):
        if instance is None:
            return self

        return self._data

    def __delete__(self, instance):
        self._data = None
        self._count = 0

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Window must be a positive integer.")

        self._window = v

    @property
    def count(self):
        return self._count


class MovingAverageScalar(_MovingAverageBase):
    """Stores moving average of a scalar number."""
    def __set__(self, instance, data):
        if data is None:
            self._data = None
            self._count = 0
            return

        if self._data is not None and self._window > 1 and \
                self._count <= self._window:
            if self._count < self._window:
                self._count += 1
                self._data += (data - self._data) / self._count
            else:  # self._count == self._window
                # here is an approximation
                self._data += (data - self._data) / self._count
        else:
            self._data = data
            self._count = 1


class MovingAverageArray(_MovingAverageBase):
    """Stores moving average of 2D/3D (and higher dimension) array data."""

    def __init__(self, window=1, *, copy_first=False):
        """Initialization.

        :param int window: moving average window size.
        :param bool copy_first: True for copy the first data.
        """
        super().__init__(window=window)

        self._copy_first = copy_first

    def __set__(self, instance, data):
        if data is None:
            self._data = None
            self._count = 0
            return

        if self._data is not None and self._window > 1 and \
                self._count <= self._window and data.shape == self._data.shape:
            if self._count < self._window:
                self._count += 1
                if data.ndim in (2, 3):
                    movingAvgImageData(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
            else:  # self._count == self._window
                # here is an approximation
                if data.ndim in (2, 3):
                    movingAvgImageData(self._data, data, self._count)
                else:
                    self._data += (data - self._data) / self._count
        else:
            self._data = data.copy() if self._copy_first else data
            self._count = 1


class SimpleQueue:
    """A thread-safe queue for passing data fast between threads.

    It does not provide the functionality of coordination among threads
    as threading.Queue, but is way more faster.
    """
    def __init__(self, maxsize=0):
        """Initialization.

        :param int maxsize: if maxsize is <= 0, the queue size is infinite.
        """
        super().__init__()

        self._queue = deque()
        self._maxsize = maxsize
        self._mutex = Lock()

    def get_nowait(self):
        """Pop an item from the queue without blocking."""
        return self.get()

    def get(self):
        with self._mutex:
            if len(self._queue) > 0:
                return self._queue.popleft()
            raise Empty

    def put_nowait(self, item):
        """Put an item into the queue without blocking."""
        self.put(item)

    def put(self, item):
        with self._mutex:
            if 0 < self._maxsize <= len(self._queue):
                raise Full
            self._queue.append(item)

    def put_pop(self, item):
        with self._mutex:
            if 0 < self._maxsize < len(self._queue):
                self._queue.popleft()
            self._queue.append(item)

    def qsize(self):
        with self._mutex:
            return len(self._queue)

    def empty(self):
        with self._mutex:
            return not len(self._queue)

    def full(self):
        with self._mutex:
            return 0 < self._maxsize <= len(self._queue)

    def clear(self):
        with self._mutex:
            self._queue.clear()
