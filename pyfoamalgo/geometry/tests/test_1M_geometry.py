import os.path as osp

import pytest

import numpy as np

import extra_geom

from pyfoamalgo.geometry import DSSC_1MGeometry, LPD_1MGeometry, AGIPD_1MGeometry
from pyfoamalgo.geometry.geometry_utils import StackView

_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../")

_IMAGE_DTYPE = np.float32
_RAW_IMAGE_DTYPE = np.uint16


class _Test1MGeometryMixin:
    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, bool])
    def testAssemblingNoPulse(self, dtype):
        modules = np.ones((self.n_modules, *self.module_shape), dtype=dtype)

        out_stack = self.geom_stack.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
        self.geom_stack.position_all_modules(modules, out_stack)

        assert (1024, 1024) == out_stack.shape[-2:]

        out_fast = self.geom_fast.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
        self.geom_fast.position_all_modules(modules, out_fast)

        out_gt = self.geom.output_array_for_position_fast(dtype=_IMAGE_DTYPE)
        self.geom.position_all_modules(modules, out_gt)

        assert out_gt.shape == out_fast.shape
        np.testing.assert_array_equal(out_fast, out_gt)

        # test dismantle
        dismantled_out = self.geom_fast.output_array_for_dismantle_fast(dtype=_IMAGE_DTYPE)
        self.geom_fast.dismantle_all_modules(out_fast, dismantled_out)

        np.testing.assert_array_equal(modules, dismantled_out)

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, bool])
    def testAssemblingArray(self, dtype):
        modules = np.ones((self.n_pulses, self.n_modules, *self.module_shape), dtype=dtype)

        out_stack = self.geom_stack.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom_stack.position_all_modules(modules, out_stack)

        assert (1024, 1024) == out_stack.shape[-2:]

        out_fast = self.geom_fast.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom_fast.position_all_modules(modules, out_fast)

        out_gt = self.geom.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom.position_all_modules(modules, out_gt)

        assert out_gt.shape == out_fast.shape
        np.testing.assert_array_equal(out_fast, out_gt)

        # test dismantle
        dismantled_out = self.geom_fast.output_array_for_dismantle_fast((self.n_pulses,), dtype=_IMAGE_DTYPE)
        self.geom_fast.dismantle_all_modules(out_fast, dismantled_out)

        np.testing.assert_array_equal(modules, dismantled_out)

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE, bool])
    def testAssemblingVector(self, dtype):
        modules = StackView(
            {i: np.ones((self.n_pulses, *self.module_shape), dtype=dtype) for i in range(self.n_modules)},
            self.n_modules,
            (self.n_pulses, ) + tuple(self.module_shape),
            dtype,
            np.nan)

        out_stack = self.geom_stack.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom_stack.position_all_modules(modules, out_stack)

        assert (1024, 1024) == out_stack.shape[-2:]

        out_fast = self.geom_fast.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom_fast.position_all_modules(modules, out_fast)

        out_gt = self.geom.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        self.geom.position_all_modules(modules, out_gt)

        assert out_gt.shape == out_fast.shape
        np.testing.assert_equal(out_fast, out_gt)

    @pytest.mark.parametrize("dtype", [_IMAGE_DTYPE, _RAW_IMAGE_DTYPE])
    def testAssemblingArrayWithTileEdgeIgnored(self, dtype):
        modules = np.ones((self.n_pulses, self.n_modules, *self.module_shape), dtype=dtype)

        out_stack = self.geom_stack.output_array_for_position_fast((self.n_pulses,), _IMAGE_DTYPE)
        # TODO: test ignore_tile_edge = False
        self.geom_stack.position_all_modules(modules, out_stack, ignore_tile_edge=True)

        th, tw = self.tile_shape[0], self.tile_shape[1]
        assert 0 == np.count_nonzero(~np.isnan(out_stack[:, :, 0::tw]))
        assert 0 == np.count_nonzero(~np.isnan(out_stack[:, :, tw - 1::tw]))
        assert 0 == np.count_nonzero(~np.isnan(out_stack[:, 0::th, :]))
        assert 0 == np.count_nonzero(~np.isnan(out_stack[:, th - 1::th, :]))


class TestDSSC_1MGeometry(_Test1MGeometryMixin):
    @classmethod
    def setup_class(cls):
        cls.geom_file = osp.join(_geom_path, "dssc_geo_june19.h5")
        quad_positions = [
            [-124.100,    3.112],
            [-133.068, -110.604],
            [   0.988, -125.236],
            [   4.528,   -4.912]
        ]
        cls.geom_stack = DSSC_1MGeometry()
        cls.geom_fast = DSSC_1MGeometry.from_h5_file_and_quad_positions(
            cls.geom_file, quad_positions)
        cls.geom = extra_geom.DSSC_1MGeometry.from_h5_file_and_quad_positions(
            cls.geom_file, quad_positions)

        cls.n_pulses = 2
        cls.n_modules = DSSC_1MGeometry.n_modules
        cls.module_shape = DSSC_1MGeometry.module_shape
        cls.tile_shape = DSSC_1MGeometry.tile_shape


class TestLPD_1MGeometry(_Test1MGeometryMixin):
    @classmethod
    def setup_class(cls):
        geom_file = osp.join(_geom_path, "lpd_mar_18_axesfixed.h5")
        quad_positions = [
            [ 11.4, 299],
            [-11.5,   8],
            [254.5, -16],
            [278.5, 275]
        ]
        cls.geom_stack = LPD_1MGeometry()
        cls.geom_fast = LPD_1MGeometry.from_h5_file_and_quad_positions(
            geom_file, quad_positions)
        cls.geom = extra_geom.LPD_1MGeometry.from_h5_file_and_quad_positions(
            geom_file, quad_positions)

        cls.n_pulses = 2
        cls.n_modules = LPD_1MGeometry.n_modules
        cls.module_shape = LPD_1MGeometry.module_shape
        cls.tile_shape = LPD_1MGeometry.tile_shape


class TestAGIPD_1MGeometry(_Test1MGeometryMixin):
    @classmethod
    def setup_class(cls):
        geom_file = osp.join(_geom_path, "agipd_mar18_v11.geom")

        cls.geom_stack = AGIPD_1MGeometry()
        cls.geom_fast = AGIPD_1MGeometry.from_crystfel_geom(geom_file)
        cls.geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(geom_file)

        cls.n_pulses = 2
        cls.n_modules = AGIPD_1MGeometry.n_modules
        cls.module_shape = AGIPD_1MGeometry.module_shape
        cls.tile_shape = AGIPD_1MGeometry.tile_shape
