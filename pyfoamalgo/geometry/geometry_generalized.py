"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""
import numpy as np

from ..lib.geometry import EPix100Geometry as _EPix100GeometryCpp
from ..lib.geometry import JungFrauGeometry as _JungFrauGeometryCpp
from .geometry_base import _GeneralizedGeometryMixin
from .geometry_utils import use_doc


class JungFrauGeometry(_GeneralizedGeometryMixin, _JungFrauGeometryCpp):
    """JungFrau geometry."""
    @classmethod
    @use_doc(_GeneralizedGeometryMixin)
    def from_crystfel_geom(cls, filename, n_rows, n_columns, *, module_numbers=None):
        """Override."""
        from cfelpyutils.crystfel_utils import load_crystfel_geometry
        from extra_geom.detectors import GeometryFragment

        if module_numbers is None:
            module_numbers = range(1, n_rows * n_columns + 1)
        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for i_p in module_numbers:
            i_a = 1 if i_p > 4 else 8
            d = geom_dict['panels'][f'p{i_p}a{i_a}']
            modules.append(GeometryFragment.from_panel_dict(d).corner_pos)
        return cls(n_rows, n_columns, modules)


class EPix100Geometry(_GeneralizedGeometryMixin, _EPix100GeometryCpp):
    """EPix100 geometry."""
    ...
