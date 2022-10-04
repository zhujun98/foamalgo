"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Author: Jun Zhu
"""
import numpy as np


__ALL_DTYPES__ = (
    np.int32, np.uint32, np.int64, np.uint64, np.float32, np.float64
)

__XFEL_IMAGE_DTYPE__ = np.float32
__XFEL_RAW_IMAGE_DTYPE__ = np.uint16

__NAN_DTYPES__ = (
    np.float32, np.float64
)
