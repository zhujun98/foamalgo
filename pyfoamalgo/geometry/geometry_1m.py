"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""
from itertools import product

import numpy as np
import h5py

from ..lib.geometry_1m import AGIPD_1MGeometry as _AGIPD_1MGeometryCpp
from ..lib.geometry_1m import LPD_1MGeometry as _LPD_1MGeometryCpp
from ..lib.geometry_1m import DSSC_1MGeometry as _DSSC_1MGeometryCpp
from .geometry_base import _1MGeometryMixin
from .geometry_utils import use_doc


_IMAGE_DTYPE = np.float32


class DSSC_1MGeometry(_1MGeometryMixin, _DSSC_1MGeometryCpp):
    """Geometry for DSSC 1M."""
    @classmethod
    @use_doc(_1MGeometryMixin)
    def from_h5_file_and_quad_positions(cls, filepath, positions):
        """Override."""
        modules = []
        with h5py.File(filepath, 'r') as f:
            for Q, M in product(range(1, cls.n_quads + 1),
                                range(1, cls.n_modules_per_quad + 1)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                # Which way round is this quadrant
                x_orient = cls.quad_orientations[Q - 1][0]
                y_orient = cls.quad_orientations[Q - 1][1]

                tiles = []
                for T in range(1, cls.n_tiles_per_module+1):
                    first_pixel_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    # mm -> m
                    first_pixel_pos[:2] = 0.001 * (quad_pos + mod_offset + tile_offset)

                    # Corner position is measured at low-x, low-y corner (bottom
                    # right as plotted). We want the position of the corner
                    # with the first pixel, which is either high-x low-y or
                    # low-x high-y.
                    if x_orient == 1:
                        first_pixel_pos[1] += cls.pixelSize()[1] * cls.tile_shape[0]
                    if y_orient == 1:
                        first_pixel_pos[0] += cls.pixelSize()[0] * cls.tile_shape[1]

                    tiles.append(list(first_pixel_pos))
                modules.append(tiles)

        return cls(modules)


class LPD_1MGeometry(_1MGeometryMixin, _LPD_1MGeometryCpp):
    """Geometry for LPD 1M."""
    @classmethod
    @use_doc(_1MGeometryMixin)
    def from_h5_file_and_quad_positions(cls, filepath, positions):
        """Override."""
        modules = []
        with h5py.File(filepath, 'r') as f:
            for Q, M in product(range(1, cls.n_quads + 1),
                                range(1, cls.n_modules_per_quad + 1)):
                quad_pos = np.array(positions[Q - 1])
                mod_grp = f['Q{}/M{}'.format(Q, M)]
                mod_offset = mod_grp['Position'][:2]

                tiles = []
                for T in range(1, cls.n_tiles_per_module+1):
                    first_pixel_pos = np.zeros(3)
                    tile_offset = mod_grp['T{:02}/Position'.format(T)][:2]
                    # mm -> m
                    first_pixel_pos[:2] = 0.001 * (quad_pos + mod_offset + tile_offset)

                    # LPD geometry is measured to the last pixel of each tile.
                    # Subtract tile dimensions for the position of 1st pixel.
                    first_pixel_pos[0] -= cls.pixelSize()[0] * cls.tile_shape[1]
                    first_pixel_pos[1] -= cls.pixelSize()[1] * cls.tile_shape[0]

                    tiles.append(list(first_pixel_pos))
                modules.append(tiles)

        return cls(modules)


class AGIPD_1MGeometry(_1MGeometryMixin, _AGIPD_1MGeometryCpp):
    """Geometry for AGIPD 1M."""
    @classmethod
    @use_doc(_1MGeometryMixin)
    def from_crystfel_geom(cls, filename):
        """Override."""
        from cfelpyutils.crystfel_utils import load_crystfel_geometry
        from extra_geom.detectors import GeometryFragment

        geom_dict = load_crystfel_geometry(filename)
        modules = []
        for i_p in range(cls.n_modules):
            tiles = []
            modules.append(tiles)
            for i_a in range(cls.n_tiles_per_module):
                d = geom_dict['panels'][f'p{i_p}a{i_a}']
                tiles.append(GeometryFragment.from_panel_dict(d).corner_pos)

        return cls(modules)
