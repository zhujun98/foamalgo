"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""

__version__ = "0.0.3"


from .statistics import (
    hist_with_stats, nanhist_with_stats, compute_statistics,
    nanmean, nansum, nanstd, nanvar,
    quick_min_max
)

from .miscellaneous import (
    normalize_auc
)
from .sampling import down_sample, slice_curve, up_sample
from .data_structures import (
    OrderedSet, Stack, SimpleSequence, SimpleVectorSequence,
    SimplePairSequence, OneWayAccuPairSequence,
)
from .azimuthal_integration import (
    compute_q, energy2wavelength, AzimuthalIntegrator, ConcentricRingsFinder,
)

from .helpers import intersection

from .imageproc import (
    nanmean_image_data, correct_image_data, mask_image_data,
    movingAvgImageData
)

from .spectrum import (
    compute_spectrum_1d
)

from .computer_vision import (
    edge_detect, fourier_transform_2d
)
