"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu
"""
import math
import numpy as np

from .imageproc import mask_image_data, nanmeanImageArray
from .config import __NAN_DTYPES__, __ALL_DTYPES__
from pyfoamalgo.lib.statistics import nanmean as _nanmean_cpp
from pyfoamalgo.lib.statistics import nansum as _nansum_cpp
from pyfoamalgo.lib.statistics import nanstd as _nanstd_cpp
from pyfoamalgo.lib.statistics import nanvar as _nanvar_cpp
from pyfoamalgo.lib.statistics import nanmin as _nanmin_cpp
from pyfoamalgo.lib.statistics import nanmax as _nanmax_cpp
from pyfoamalgo.lib.statistics import histogram1d as _histogram1d_cpp

__all__ = [
    'hist_with_stats',
    'nanhist_with_stats',
    'compute_statistics',
    'nanmean',
    'nansum',
    'nanstd',
    'nanvar',
    'nanmin',
    'nanmax',
    'quick_min_max',
    'histogram1d',
]


def nansum(a, axis=None):
    """Faster numpy.nansum.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nansum.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the sum is computed.
        The default is to compute the sum of the flattened array.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis is None:
            return _nansum_cpp(a)
        return _nansum_cpp(a, axis=axis)

    return np.nansum(a, axis=axis)


def nanmean(a, axis=None):
    """Faster numpy.nanmean.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nanmean.

    If the input array is an array of images, i.e. 3D array, one may
    want to check :func:`pyfoamalgo.nanmean_image_data`.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the mean is computed.
        The default is to compute the mean of the flattened array.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis == 0 and a.ndim == 3:
            return nanmeanImageArray(a)
        if axis is None:
            return _nanmean_cpp(a)
        return _nanmean_cpp(a, axis=axis)

    return np.nanmean(a, axis=axis)


def nanstd(a, axis=None, *, normalized=False):
    """Faster numpy.nanstd.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nanstd.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the standard
        deviation is computed. The default is to compute the standard
        deviation of the flattened array.
    :param bool normalized: True for normalizing the result by nanmean
        along the same axis or axes.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis is None:
            ret = _nanstd_cpp(a)
        else:
            ret = _nanstd_cpp(a, axis=axis)
    else:
        ret = np.nanstd(a, axis=axis)

    if normalized:
        return ret / nanmean(a, axis=axis)
    return ret


def nanvar(a, axis=None, *, normalized=False):
    """Faster numpy.nanvar.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nanvar.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the variance
        is computed. The default is to compute the variance of the
        flattened array.
    :param bool normalized: True for normalizing the result by square of
        nanmean along the same axis or axes.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis is None:
            ret = _nanvar_cpp(a)
        else:
            ret = _nanvar_cpp(a, axis=axis)
    else:
        ret = np.nanvar(a, axis=axis)

    if normalized:
        return ret / nanmean(a, axis=axis) ** 2
    return ret


def nanmin(a, axis=None):
    """Faster numpy.nanmin.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nanmin.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the mean is computed.
        The default is to compute the nanmin of the flattened array.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis is None:
            return _nanmin_cpp(a)
        return _nanmin_cpp(a, axis=axis)

    return np.nanmin(a, axis=axis)


def nanmax(a, axis=None):
    """Faster numpy.nanmax.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.nanmax.

    :param numpy.ndarray a: Data array.
    :param None/int/tuple axis: Axis or axes along which the mean is computed.
        The default is to compute the nanmax of the flattened array.
    """
    if a.dtype in __NAN_DTYPES__:
        if axis is None:
            return _nanmax_cpp(a)
        return _nanmax_cpp(a, axis=axis)

    return np.nanmax(a, axis=axis)


def histogram1d(a, bins=10, range=None):
    """Faster numpy.histogram.

    It uses the C++ implementation when applicable. Otherwise, it falls
    back to numpy.histogram.

    :param numpy.ndarray a: Data array.
    :param int bins: Number of bins.
    :param tuple/None range: The (lower, upper) boundary of the bins.
        Default = (a.min(), a.max())

    :return: (Values of the histogram, bin edges)
    :rtype: (numpy.array, numpy.array)
    """
    if range is None:
        range = (a.min(), a.max())

    if a.dtype in __ALL_DTYPES__:
        be_dtype = np.float32 if a.dtype == np.float32 else np.float64
        bin_edges = np.linspace(range[0], range[1], bins+1, dtype=be_dtype)
        return (_histogram1d_cpp(a.ravel(), range[0], range[1], bins),
                bin_edges)
    return np.histogram(a, bins=bins, range=range)


def quick_min_max(x, q=None):
    """Estimate the min/max values of input by down-sampling.

    :param numpy.ndarray x: data, 2D array for now.
    :param float/None q: quantile when calculating the min/max, which
        must be within [0, 1].

    :return tuple: (min, max)
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray!")

    if x.ndim != 2:
        raise ValueError("Input must be a 2D array!")

    while x.size > 1e5:
        sl = [slice(None)] * x.ndim
        sl[np.argmax(x.shape)] = slice(None, None, 2)
        x = x[tuple(sl)]

    if q is None:
        return np.nanmin(x), np.nanmax(x)

    if q < 0.5:
        q = 1 - q

    # Let np.nanquantile to handle the case when q is outside [0, 1]
    # caveat: nanquantile is about 30 times slower than nanmin/nanmax
    return np.nanquantile(x, 1 - q, method='nearest'), \
           np.nanquantile(x, q, method='nearest')


def _get_outer_edges(arr, bin_range):
    """Determine the outer bin edges to use.

    From both the data and the range argument.

    :param numpy.ndarray arr: Data.
    :param tuple bin_range: Desired range (min, max).

    :return tuple: Outer edges (min, max).

    Note: the input array is assumed to be nan-free but could contain +-inf.
          The returned outer edges could be inf or -inf if both the min/max
          value of array and the corresponding boundary of the range argument
          are inf or -inf.
    """
    if bin_range is None:
        bin_range = (-math.inf, math.inf)

    v_min, v_max = bin_range
    assert v_min < v_max

    if not np.isfinite(v_min) and not np.isfinite(v_max):
        if arr.size == 0:
            v_min, v_max = 0., 0.
        else:
            v_min, v_max = np.min(arr), np.max(arr)

        if v_min == v_max:
            # np.histogram convention
            v_min -= 0.5
            v_max += 0.5
    elif not np.isfinite(v_max):
        if arr.size == 0:
            v_max = v_min + 1.0
        else:
            v_max = np.max(arr)
            if v_max <= v_min:
                # this could happen when v_max is +Inf while v_min is finite
                v_max = v_min + 1.0  # must have v_max > v_min
    elif not np.isfinite(v_min):
        if arr.size == 0:
            v_min = v_max - 1.0
        else:
            v_min = np.min(arr)
            if v_min >= v_max:
                # this could happen when v_min is -Inf while v_max is finite
                v_min = v_max - 1.0  # must have v_max > v_min

    return v_min, v_max


def compute_statistics(data):
    """Compute statistics of an array.

    :param numpy.ndarray data: Input array.
    """
    if len(data) == 0:
        # suppress runtime warning
        return np.nan, np.nan, np.nan
    return np.mean(data), np.median(data), np.std(data)


def nanhist_with_stats(data, bin_range=None, n_bins=10):
    """Compute nan-histogram and nan-statistics of an array.

    :param numpy.ndarray data: Image ROI.
    :param tuple bin_range: (lb, ub) of histogram.
    :param int n_bins: Number of bins of histogram.

    :raise ValueError: if finite outer edges cannot be found.
    """
    # Note: Since the nan functions in numpy is typically 5-8 slower
    # than the non-nan counterpart, it is always faster to remove nan
    # first, which results in a copy, and then calculate the statistics.

    # TODO: the following three steps can be merged into one to improve
    #       the performance.
    filtered = data.copy()
    mask_image_data(filtered, threshold_mask=bin_range)
    filtered = filtered[~np.isnan(filtered)]

    outer_edges = _get_outer_edges(filtered, bin_range)
    hist, bin_edges = np.histogram(filtered, range=outer_edges, bins=n_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    mean, median, std = compute_statistics(filtered)

    return hist, bin_centers, mean, median, std


def hist_with_stats(data, bin_range=None, n_bins=10):
    """Compute histogram and statistics of an array.

    :param numpy.ndarray data: Input data.
    :param tuple bin_range: (lb, ub) of histogram.
    :param int n_bins: Number of bins of histogram.

    :raise ValueError: if finite outer edges cannot be found.
    """
    v_min, v_max = _get_outer_edges(data, bin_range)

    filtered = data[(data >= v_min) & (data <= v_max)]
    hist, bin_edges = np.histogram(
        filtered, bins=n_bins, range=(v_min, v_max))
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    mean, median, std = compute_statistics(filtered)

    return hist, bin_centers, mean, median, std
