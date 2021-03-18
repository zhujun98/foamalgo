import pytest
import numpy as np

from pyfoamalgo.config import __XFEL_IMAGE_DTYPE__ as IMAGE_DTYPE
from pyfoamalgo.geometry import stack_detector_modules


class TestStackDetectorModules:
    def test_screening_input(self):
        with pytest.raises(ValueError, match="Empty data"):
            stack_detector_modules({}, "ABC", 'abc')

        with pytest.raises(ValueError, match="Device name must contain a "):
            stack_detector_modules(
                {'ABC': {'abc': np.ones((2, 2))}}, "ABC", 'abc')

        with pytest.raises(ValueError, match="At least one of"):
            stack_detector_modules(
                {'ABC1D': {'abc': np.ones((2, 2))}}, "ABC*D", 'abc')

        with pytest.raises(ValueError, match="Number of modules cannot exceed"):
            stack_detector_modules(
                {'ABC1D': {'abc': np.ones((2, 2))}}, "ABC*D", 'abc', modules=17)
        with pytest.raises(ValueError, match="Number of modules cannot exceed"):
            stack_detector_modules(
                {'ABC1D': {'abc': np.ones((2, 2))}}, "ABC*D", 'abc',
                module_numbers=np.arange(17))

    def test_stack_1M_detector(self):
        ppt_name = 'image.data'
        shape = (4, 1, 256, 256)
        dtype = IMAGE_DTYPE
        train_data = {
            'SPB_DET_AGIPD1M-1/DET/11CH0:xtdf':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'SPB_DET_AGIPD1M-1/DET/7CH0:xtdf':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'SPB_DET_AGIPD1M-1/DET/8CH0:xtdf':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'SPB_DET_AGIPD1M-1/DET/3CH0:xtdf':
                {ppt_name: np.ones(shape, dtype=dtype)},
        }

        modules_data = stack_detector_modules(
            train_data, 'SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', modules=16)
        assert shape[:2] + (16,) + shape[-2:] == modules_data.shape
        assert dtype == modules_data.dtype

        # test dtype mismatch
        train_data['SPB_DET_AGIPD1M-1/DET/2CH0:xtdf'] = {ppt_name: np.ones(shape, dtype=np.float64)}
        with pytest.raises(ValueError, match="Modules have mismatched dtypes"):
            stack_detector_modules(
                train_data, 'SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', modules=16)

        # test shape mismatch
        train_data['SPB_DET_AGIPD1M-1/DET/2CH0:xtdf'] = {ppt_name: np.ones((4, 256, 256), dtype=dtype)}
        with pytest.raises(ValueError, match="Modules have mismatched shapes"):
            stack_detector_modules(
                train_data, 'SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', modules=16)

    @pytest.mark.parametrize("memory_cell_last", [False, True])
    def test_stack_generalized_detector_ps(self, memory_cell_last):
        ppt_name = 'data.adc'
        module_shape = (512, 1024)
        n_cells = 16
        if memory_cell_last:
            shape = module_shape + (n_cells,)
        else:
            shape = (n_cells,) + module_shape
        dtype = IMAGE_DTYPE
        train_data = {
            'FXE_XAD_JF1M/DET/RECEIVER-1:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'FXE_XAD_JF1M/DET/RECEIVER-2:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'FXE_XAD_JF1M/DET/RECEIVER-3:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'FXE_XAD_JF1M/DET/RECEIVER-4:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
        }

        modules_data = stack_detector_modules(
            train_data, 'FXE_XAD_JF1M/DET/RECEIVER-*:daqOutput', ppt_name,
            modules=4, memory_cell_last=memory_cell_last)
        assert (n_cells,) + (4,) + module_shape == modules_data.shape
        assert dtype == modules_data.dtype

    @pytest.mark.parametrize("memory_cell_last", [False, True])
    def test_stack_generalized_detector_ts(self, memory_cell_last):
        ppt_name = 'data.image.pixels'
        shape = (708, 768)
        dtype = IMAGE_DTYPE
        train_data = {
            'MID_EXP_EPIX-1/DET/RECEIVER:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
            'MID_EXP_EPIX-2/DET/RECEIVER:daqOutput':
                {ppt_name: np.ones(shape, dtype=dtype)},
        }

        modules_data = stack_detector_modules(
            train_data, 'MID_EXP_EPIX-*/DET/RECEIVER:daqOutput', ppt_name, modules=2)
        assert (2,) + shape[-2:] == modules_data.shape
        assert dtype == modules_data.dtype
