"""
Distributed under the terms of the BSD 3-Clause License.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu <jun.zhu@xfel.eu>
Copyright (C) European X-Ray Free-Electron Laser Facility GmbH.
All rights reserved.
"""
import numpy as np
from scipy import fft

from .imageproc import mask_image_data
from pyfoamalgo.lib.canny import cannyEdge
from pyfoamalgo.lib.smooth import gaussianBlur

__all__ = [
    'edge_detect',
    'fourier_transform_2d'
]


def edge_detect(image, *,
                kernel_size=3, sigma=1, threshold=(0, 1),
                mask_nan=True):
    """Detect edges in an image.

    :param numpy.ndarray image: image data. Shape = (y, x)
    :param int kernel_size: kernel size for Gaussian blur.
    :param float sigma: Gaussian kernel standard deviation.
    :param tuple threshold: (first, second) thresholds for the hysteresis
        procedure.
    :param bool mask_nan: whether to mask nan values to 0.
    """
    masked = image
    if mask_nan:
        masked = np.copy(image)
        mask_image_data(masked, keep_nan=False)

    blurred = np.zeros_like(masked)
    gaussianBlur(masked, blurred, kernel_size, sigma)
    out = np.zeros_like(blurred, dtype=np.uint8)
    cannyEdge(blurred, out, threshold[0], threshold[1])

    return out


def fourier_transform_2d(image, *, logrithmic=True, mask_nan=True):
    """Compute the 2-dimensional discrete Fourier Transform.

    :param numpy.ndarray image: image data. Shape = (y, x)
    :param logrithmic: True for returning logrithmic values of the real part
        of the transform.
    :param bool mask_nan: whether to mask nan values to 0.

    :return numpy.ndarray: transformed image data. Shape = (y, x)
    """
    masked = np.copy(image)
    if mask_nan:
        mask_image_data(masked, keep_nan=False)

    # TODO: improve performance
    out = fft.fftshift(fft.fft2(masked, overwrite_x=True))

    np.abs(out, out=out)
    if logrithmic:
        np.log10(1 + out, out=out)
    return out
