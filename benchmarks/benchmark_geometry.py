import os.path as osp
import time

import numpy as np

from pyfoamalgo.config import __XFEL_IMAGE_DTYPE__ as IMAGE_DTYPE
from pyfoamalgo.config import __XFEL_RAW_IMAGE_DTYPE__ as RAW_IMAGE_DTYPE

_data_sources = [(RAW_IMAGE_DTYPE, 'raw'),
                 (IMAGE_DTYPE, 'calibrated')]

_geom_path = osp.join(osp.dirname(osp.abspath(__file__)), "../pyfoamalgo/geometry")


def _benchmark_1m_imp(geom_fast_cls, geom_cls, geom_file,
                      quad_positions=None,
                      n_pulses=32):

    for from_dtype, from_str in _data_sources:
        modules = np.ones((n_pulses,
                           geom_fast_cls.n_modules,
                           geom_fast_cls.module_shape[0],
                           geom_fast_cls.module_shape[1]), dtype=from_dtype)

        # assemble with geometry and quad position in EXtra-geom

        if quad_positions is not None:
            geom = geom_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_cls.from_crystfel_geom(geom_file)
        assembled = geom.output_array_for_position_fast((n_pulses,), dtype=IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, out=assembled)
        dt_geom = time.perf_counter() - t0

        # stack only

        geom = geom_fast_cls()
        assembled = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_foam_stack = time.perf_counter() - t0

        # assemble with geometry and quad position

        if quad_positions is not None:
            geom = geom_fast_cls.from_h5_file_and_quad_positions(geom_file, quad_positions)
        else:
            geom = geom_fast_cls.from_crystfel_geom(geom_file)
        assembled = np.full((n_pulses, *geom.assembledShape()), np.nan, dtype=IMAGE_DTYPE)
        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_foam = time.perf_counter() - t0

        print(f"\nposition all modules for {geom_cls.__name__} (from {from_str} data) - \n"
              f"  dt (foam stack only): {dt_foam_stack:.4f}, dt (foam): {dt_foam:.4f}, "
              f"dt (geom): {dt_geom:.4f}")

        if modules.dtype == IMAGE_DTYPE:
            t0 = time.perf_counter()
            geom.dismantle_all_modules(assembled, modules)
            dt_foam_dismantle = time.perf_counter() - t0

            print(f"\ndismantle all modules for {geom_cls.__name__} (from {from_str} data) - \n"
                  f"  dt (foam): {dt_foam_dismantle:.4f}")


def benchmark_dssc_1m():
    from pyfoamalgo.geometry import DSSC_1MGeometry as DSSC_1MGeometryFast
    from extra_geom import DSSC_1MGeometry

    geom_file = osp.join(_geom_path, "dssc_geo_june19.h5")
    quad_positions = [
        [-124.100,    3.112],
        [-133.068, -110.604],
        [   0.988, -125.236],
        [   4.528,   -4.912]
    ]

    _benchmark_1m_imp(DSSC_1MGeometryFast, DSSC_1MGeometry, geom_file, quad_positions)


def benchmark_lpd_1m():
    from pyfoamalgo.geometry import LPD_1MGeometry as LPD_1MGeometryFast
    from extra_geom import LPD_1MGeometry

    geom_file = osp.join(_geom_path, "lpd_mar_18_axesfixed.h5")
    quad_positions = [
        [ 11.4, 299],
        [-11.5,   8],
        [254.5, -16],
        [278.5, 275]
    ]

    _benchmark_1m_imp(LPD_1MGeometryFast, LPD_1MGeometry, geom_file, quad_positions)


def benchmark_agipd_1m():
    from pyfoamalgo.geometry import AGIPD_1MGeometry as AGIPD_1MGeometryFast
    from extra_geom import AGIPD_1MGeometry

    geom_file = osp.join(_geom_path, "agipd_mar18_v11.geom")

    _benchmark_1m_imp(AGIPD_1MGeometryFast, AGIPD_1MGeometry, geom_file)


def benchmark_jungfrau():
    from pyfoamalgo.geometry import JungFrauGeometry as JungFrauGeometryFast

    for from_dtype, from_str in _data_sources:
        n_row, n_col = 3, 2
        geom = JungFrauGeometryFast(n_row, n_col)
        n_pulses = 16
        modules = np.ones((n_pulses, n_row * n_col, *geom.module_shape), dtype=from_dtype)

        assembled = geom.output_array_for_position_fast((n_pulses,), IMAGE_DTYPE)

        t0 = time.perf_counter()
        geom.position_all_modules(modules, assembled)
        dt_assemble = time.perf_counter() - t0

        print(f"\nposition all modules for JungFrauGeometry (from {from_str} data) - \n"
              f"  dt (foam stack only): {dt_assemble:.4f}")

        if modules.dtype == IMAGE_DTYPE:
            t0 = time.perf_counter()
            geom.dismantle_all_modules(assembled, modules)
            dt_dismantle = time.perf_counter() - t0

            print(f"\ndismantle all modules for JungFrauGeometry (from {from_str} data) - \n"
                  f"  dt (foam stack only): {dt_dismantle:.4f}")

    module = np.ones((n_pulses, *geom.module_shape), dtype=IMAGE_DTYPE)
    t0 = time.perf_counter()
    JungFrauGeometryFast.maskModule(module)
    dt_mask_cpp = time.perf_counter() - t0

    def _mask_module_py(module):
        ah, aw = JungFrauGeometryFast.asic_shape
        ny, nx = JungFrauGeometryFast.asic_grid_shape
        for i in range(ny):
            module[..., i * ah, :] = np.nan
            module[..., (i + 1) * ah - 1, :] = np.nan
        for j in range(nx):
            module[..., :, j * aw] = np.nan
            module[..., :, (j + 1) * aw - 1] = np.nan

    module = np.ones((n_pulses, *geom.module_shape), dtype=IMAGE_DTYPE)
    t0 = time.perf_counter()
    _mask_module_py(module)
    dt_mask_py = time.perf_counter() - t0

    print(f"\nMask single module for JungFrauGeometry - \n"
          f"  dt (cpp): {dt_mask_cpp:.4f}, dt (py): {dt_mask_py:.4f}")


if __name__ == "__main__":
    print("*" * 80)
    print("Benchmark geometry")
    print("*" * 80)

    benchmark_dssc_1m()

    benchmark_lpd_1m()

    benchmark_agipd_1m()

    benchmark_jungfrau()
