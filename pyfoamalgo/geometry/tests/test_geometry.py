import os.path as osp

import pytest

import numpy as np

from pyfoamalgo.config import __XFEL_IMAGE_DTYPE__ as IMAGE_DTYPE
from pyfoamalgo.config import __XFEL_RAW_IMAGE_DTYPE__ as RAW_IMAGE_DTYPE
from pyfoamalgo.geometry import EPix100Geometry, JungFrauGeometry
from pyfoamalgo.geometry.geometry_utils import StackView

_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../")


class TestJungFrauGeometry:
    """Test pulse-resolved."""
    @classmethod
    def setup_class(cls):
        cls.n_pulses = 2
        cls.module_shape = JungFrauGeometry.module_shape
        cls.asic_shape = JungFrauGeometry.asic_shape
        cls.pixel_size = JungFrauGeometry.pixel_size

        cls.geom_21_stack = JungFrauGeometry(2, 1)
        cls.geom_32_stack = JungFrauGeometry(3, 2)

        cls.cases = [
            (cls.geom_21_stack, 2, (1024, 1024)),
            (cls.geom_32_stack, 6, (1536, 2048)),
        ]

        # TODO: add default JungFrau geometries
        geom_file = osp.join(osp.expanduser("~"), "jungfrau.geom")
        try:
            cls.geom_32_cfel = JungFrauGeometry.from_crystfel_geom(
                geom_file, n_rows=3, n_columns=2, module_numbers=[1, 2, 3, 6, 7, 8])
        except FileNotFoundError:
            module_coordinates = [
                np.array([ 0.08452896,  0.07981445, 0.]),
                np.array([ 0.08409096,  0.03890507, 0.]),
                np.array([ 0.08385471, -0.00210121, 0.]),
                np.array([-0.08499321, -0.04048030, 0.]),
                np.array([-0.08477046,  0.00059965, 0.]),
                np.array([-0.08479671,  0.04162323, 0.])
            ]

            cls.geom_32_cfel = JungFrauGeometry(3, 2, module_coordinates)
        cls.cases.append((cls.geom_32_cfel, 6, (1607, 2260)))

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, RAW_IMAGE_DTYPE),
                              (bool, bool)])
    def testAssemblingArray(self, src_dtype, dst_dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((self.n_pulses, n_modules, *self.module_shape), dtype=src_dtype)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), dst_dtype)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

            # test dismantle
            dismantled = geom.output_array_for_dismantle_fast((self.n_pulses,), dst_dtype)
            geom.dismantle_all_modules(assembled, dismantled)
            np.testing.assert_array_equal(modules, dismantled)

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, RAW_IMAGE_DTYPE),
                              (bool, bool)])
    def testAssemblingVector(self, src_dtype, dst_dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = StackView(
                {i: np.ones((self.n_pulses, *self.module_shape), dtype=src_dtype) for i in range(n_modules)},
                n_modules,
                (self.n_pulses, ) + tuple(self.module_shape),
                src_dtype,
                np.nan)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), dst_dtype)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE)])
    def testAssemblingArrayWithAsicEdgeIgnored(self, src_dtype, dst_dtype):
        # the destination array must have a floating point dtype which allows nan

        ah, aw = self.asic_shape[0], self.asic_shape[1]

        # assembling with a geometry file is not tested
        for geom, n_modules, assembled_shape_gt in self.cases[:-1]:
            modules = np.ones((self.n_pulses, n_modules, *self.module_shape), dtype=src_dtype)

            assembled = geom.output_array_for_position_fast((self.n_pulses,), dst_dtype)
            geom.position_all_modules(modules, assembled, ignore_asic_edge=True)

            assert 0 == np.count_nonzero(~np.isnan(assembled[:, :, 0::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, :, aw - 1::aw]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, 0::ah, :]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[:, ah - 1::ah, :]))

    def testMaskModule(self):
        module1 = np.ones((self.n_pulses, *self.module_shape), dtype=IMAGE_DTYPE)
        module2 = np.copy(module1)

        JungFrauGeometry.maskModule(module1)

        ah, aw = JungFrauGeometry.asic_shape
        ny, nx = JungFrauGeometry.asic_grid_shape
        for i in range(ny):
            module2[..., i * ah, :] = np.nan
            module2[..., (i + 1) * ah - 1, :] = np.nan
        for j in range(nx):
            module2[..., :, j * aw] = np.nan
            module2[..., :, (j + 1) * aw - 1] = np.nan

        np.testing.assert_array_equal(module1, module2)


class TestEPix100Geometry:
    """Test train-resolved."""
    @classmethod
    def setup_class(cls):
        cls.module_shape = EPix100Geometry.module_shape
        cls.asic_shape = EPix100Geometry.asic_shape
        cls.pixel_size = EPix100Geometry.pixel_size

        cls.geom_21_stack = EPix100Geometry(2, 1)
        cls.geom_22_stack = EPix100Geometry(2, 2)

        cls.cases = [
            (cls.geom_21_stack, 2, (1416, 768)),
            (cls.geom_22_stack, 4, (1416, 1536)),
        ]

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE),
                              (np.int16, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, RAW_IMAGE_DTYPE),
                              (bool, bool)])
    def testAssemblingArray(self, src_dtype, dst_dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((n_modules, *self.module_shape), dtype=src_dtype)

            assembled = geom.output_array_for_position_fast(dtype=dst_dtype)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

            # test dismantle
            dismantled = geom.output_array_for_dismantle_fast(dtype=dst_dtype)
            geom.dismantle_all_modules(assembled, dismantled)
            np.testing.assert_array_equal(modules, dismantled)

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE),
                              (np.int16, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, RAW_IMAGE_DTYPE),
                              (bool, bool)])
    def testAssemblingVector(self, src_dtype, dst_dtype):
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = StackView(
                {i: np.ones(self.module_shape, dtype=src_dtype) for i in range(n_modules)},
                n_modules,
                tuple(self.module_shape),
                src_dtype,
                np.nan)

            assembled = geom.output_array_for_position_fast(dtype=dst_dtype)
            geom.position_all_modules(modules, assembled)
            assert assembled_shape_gt == assembled.shape[-2:]

    @pytest.mark.parametrize("src_dtype,dst_dtype",
                             [(IMAGE_DTYPE, IMAGE_DTYPE),
                              (RAW_IMAGE_DTYPE, IMAGE_DTYPE),
                              (np.int16, IMAGE_DTYPE)])
    def testAssemblingWithAsicEdgeIgnored(self, src_dtype, dst_dtype):
        # the destination array must have a floating point dtype which allows nan

        mh, mw = self.module_shape[0], self.module_shape[1]
        for geom, n_modules, assembled_shape_gt in self.cases:
            modules = np.ones((n_modules, *self.module_shape), dtype=src_dtype)

            assembled = geom.output_array_for_position_fast(dtype=dst_dtype)
            geom.position_all_modules(modules, assembled, ignore_asic_edge=True)
            assert n_modules * mw * 2 == np.count_nonzero(np.isnan(assembled))
            assert 0 == np.count_nonzero(~np.isnan(assembled[0::mh, :]))
            assert 0 == np.count_nonzero(~np.isnan(assembled[mh - 1::mh, :]))

    def testMaskModule(self):
        module1 = np.ones(self.module_shape, dtype=IMAGE_DTYPE)
        module2 = np.copy(module1)

        EPix100Geometry.maskModule(module1)

        module2[0, :] = np.nan
        module2[-1, :] = np.nan

        np.testing.assert_array_equal(module1, module2)
