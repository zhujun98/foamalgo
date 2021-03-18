"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>, Ebad Kamil <ebad.kamil@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np

from pyfoamalgo.lib.imageproc import (
    nanmeanImageArray,
    imageDataNanMask, maskImageDataNan, maskImageDataZero,
    correctGain, correctOffset, correctDsscOffset, correctGainOffset
)

__all__ = [
    'nanmean_image_data',
    'nanmean_images',
    'correct_image_data',
    'mask_image_data',
]


def nanmean_image_data(data, *, kept=None):
    """Compute nanmean of an array of images of a tuple/list of two images.

    :param numpy.array data: a 2D or 3D array. If the input is a 2D array, a
        copy will be returned. This seemingly awkward 'feature' is a sugar for
        having clean code in EXtra-foam in order to deal train- and
        pulse-resolved detectors at the same time.
    :param None/list kept: Indices of the kept images.

    :return: nanmean of the input data.
    :rtype: numpy.ndarray.
    """
    if data.ndim == 2:
        return data.copy()

    if kept is None:
        return nanmeanImageArray(data)

    return nanmeanImageArray(data, kept)


def nanmean_images(image1, image2):
    """Compute nanmean of two images.

    There is no copy overhead.

    :param numpy.array image1: The first image, Shape = (y, x).
    :param numpy.array image2: The second image, Shape = (y, x).

    :return: nanmean of the two input images.
    :rtype: numpy.ndarray.
    """
    return nanmeanImageArray(image1, image2)


def correct_image_data(data, *,
                       gain=None,
                       offset=None,
                       intradark=False,
                       detector=""):
    """Apply gain and/or offset correct to image data.

    :param numpy.array data: image data, Shape = (y, x) or (indices, y, x)
    :param None/numpy.array gain: Gain constants, which has the same
        shape as the image data.
    :param None/numpy.array offset: Offset constants, which has the same
        shape as the image data.
    :param bool intradark: Apply interleaved intra-dark correction after
        the gain/offset correction. In other words, for every other image
        in the array starting from the first one, it will be subtracted
        by the image next to it.
    :param str detector: Detector name. If given, specialized correction
        may be applied. "DSSC" - change data pixels with value 0 to 256
        before applying offset correction.
    """
    if gain is not None and offset is not None:
        correctGainOffset(data, gain, offset)
    elif offset is not None:
        if detector == "DSSC":
            correctDsscOffset(data, offset)
        else:
            correctOffset(data, offset)
    elif gain is not None:
        correctGain(data, gain)

    if intradark:
        correctOffset(data)


def mask_image_data(data, *,
                    image_mask=None,
                    threshold_mask=None,
                    keep_nan=True,
                    out=None):
    """Mask image data by image mask and/or threshold mask.

    :param numpy.ndarray data: Image data to be masked.
        Shape = (y, x) or (indices, y, x)
    :param numpy.ndarray image_mask: Image mask. If provided, it must have
        the same shape as a single image, and the type must be bool.
        Shape = (y, x)
    :param tuple/None threshold_mask: (min, max) of the threshold mask.
    :param bool keep_nan: True for masking all pixels in nan and False for
        masking all pixels to zero.
    :param numpy.ndarray out: Optional output array in which to mark the
        union of all pixels being masked. The default is None; if provided,
        it must have the same shape as the image, and the dtype must be bool.
        Only available if the image data is a 2D array. Shape = (y, x)
    """
    f = maskImageDataNan if keep_nan else maskImageDataZero

    if out is None:
        if image_mask is None and threshold_mask is None:
            f(data)
        elif image_mask is None:
            f(data, *threshold_mask)
        elif threshold_mask is None:
            f(data, image_mask)
        else:
            f(data, image_mask, *threshold_mask)
    else:
        if data.ndim == 3:
            raise ValueError("'arr' must be 2D when 'out' is specified!")

        if out.dtype != bool:
            raise ValueError("Type of 'out' must be bool!")

        if image_mask is None:
            if threshold_mask is None:
                imageDataNanMask(data, out)  # get the mask
                f(data)  # mask nan (only for keep_nan = False)
            else:
                f(data, *threshold_mask, out)
        else:
            if threshold_mask is None:
                f(data, image_mask, out)
            else:
                f(data, image_mask, *threshold_mask, out)
