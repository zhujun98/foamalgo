"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) 2020, Jun Zhu. All rights reserved.
"""
import abc

import numpy as np

from .geometry_utils import use_doc


_IMAGE_DTYPE = np.float32


class _GeometryMixin:
    """Mixin class for 1M and generalized geometry.

    The mixin class implements the API methods which have the same signatures
    as those implemented in EXtra-geom.
    """
    def output_array_for_position_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of assembled data filled with nan.

        :param tuple extra_shape: By default, a 2D array is generated to hold
            the assembling image of modules from a single pulse. For
            assembling multiple pulses at once, pass ``extra_shape=(pulses,)``
            to return a 3D array.
        :param numpy.dtype dtype: dtype of the output array.
        """
        shape = extra_shape + tuple(self.assembledShape())
        if dtype == np.bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    @abc.abstractmethod
    def position_all_modules(self, modules, out, *,
                             ignore_tile_edge=False, ignore_asic_edge=False):
        """Assemble data in modules according to where the pixels are.

        :param numpy.ndarray/list modules: data in modules.
            Shape = (memory cells, modules, y x) / (modules, y, x)
        :param numpy.ndarray out: assembled data.
            Shape = (memory cells, y, x) / (y, x)
        :param ignore_tile_edge: True for ignoring the pixels at the edges
            of tiles. If 'out' is pre-filled with nan, it it equivalent to
            masking the tile edges.
        :param ignore_asic_edge: True for ignoring the pixels at the edges
            of asics. If 'out' is pre-filled with nan, it it equivalent to
            masking the asic edges.
        """
        pass

    @abc.abstractmethod
    def output_array_for_dismantle_fast(self, extra_shape=(), dtype=_IMAGE_DTYPE):
        """Make an array with the shape of data in modules filled with nan.

        :param tuple extra_shape: By default, a 3D array is generated to hold
            the dismantled modules from a single assembled image. For
            dismantling multiple images at once, pass ``extra_shape=(pulses,)``
            to return a 4D array.
        :param numpy.dtype dtype: dtype of the output array.
        """
        shape = extra_shape + (self.n_modules, *self.module_shape)
        if dtype == np.bool:
            return np.full(shape, 0, dtype=dtype)
        return np.full(shape, np.nan, dtype=dtype)

    def dismantle_all_modules(self, assembled, out):
        """Dismantle assembled data into data in modules.

        :param numpy.ndarray out: assembled data.
            Shape = (memory cells, y, x) / (y, x)
        :param numpy.ndarray out: data in modules.
            Shape = (memory cells, modules, y x) / (modules, y, x)
        """
        self.dismantleAllModules(assembled, out)


class _1MGeometryMixin(_GeometryMixin):
    """Mixin class for 1M geometry."""

    @classmethod
    def from_h5_file_and_quad_positions(cls, filepath, positions):
        """Construct a geometry from an XFEL HDF5 format geometry file.

        :param str filename: Path of the geometry file.
        :param list positions: a list of 4 (x, y) coordinates of the
            corner of each quadrant.
        """
        raise NotImplementedError

    @classmethod
    def from_crystfel_geom(cls, filename):
        """Construct a geometry from an CrystFEL format geometry file.

        :param str filename: Path of the geometry file.
        """
        raise NotImplementedError

    @use_doc(_GeometryMixin)
    def position_all_modules(self, modules, out, *,
                             ignore_tile_edge=False, ignore_asic_edge=False):
        """Override."""
        if ignore_asic_edge:
            raise NotImplementedError(
                "1M Geometry does not support masking ASIC edges")

        if isinstance(modules, np.ndarray):
            self.positionAllModules(modules, out, ignore_tile_edge)
        else:  # extra_data.StackView
            ml = []
            for i in range(self.n_modules):
                ml.append(modules[:, i, ...])
            self.positionAllModules(ml, out, ignore_tile_edge)


class _GeneralizedGeometryMixin(_GeometryMixin):
    """Mixin class for generalized geometry."""
    @classmethod
    def from_crystfel_geom(cls, filename, n_rows, n_columns, *, module_numbers=None):
        """Construct a geometry from an CrystFEL format geometry file.

        :param str filename: Path of the geometry file.
        :param int n_rows: Number of rows of the grid layout.
        :param int n_columns: Number of columns of the grid layout.
        :param list module_numbers: A list of module numbers.
        """
        raise NotImplementedError

    @property
    def n_modules(self):
        return self.nModules()

    @use_doc(_GeometryMixin)
    def position_all_modules(self, modules, out, *,
                             ignore_tile_edge=False, ignore_asic_edge=False):
        """Override."""
        if ignore_tile_edge:
            raise NotImplementedError(
                "Generalized Geometry does not support masking tile edges")

        if isinstance(modules, np.ndarray):
            self.positionAllModules(modules, out, ignore_asic_edge)
        else:  # extra_data.StackView
            ml = []
            for i in range(self.nModules()):
                ml.append(modules[..., i, :, :])
            self.positionAllModules(ml, out, ignore_asic_edge)
