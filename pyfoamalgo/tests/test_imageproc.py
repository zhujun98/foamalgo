import pytest

import warnings

import numpy as np

from pyfoamalgo.config import __XFEL_IMAGE_DTYPE__ as IMAGE_DTYPE
from pyfoamalgo.config import __NAN_DTYPES__
from pyfoamalgo import (
    correct_image_data, mask_image_data, nanmean_image_data
)
from pyfoamalgo.lib.imageproc import movingAvgImageData


class TestImageProc:
    @pytest.mark.parametrize("dtype", __NAN_DTYPES__)
    def testNanmeanImageData(self, dtype):
        arr1d = np.ones(2, dtype=dtype)
        arr2d = np.ones((2, 2), dtype=dtype)
        arr3d = np.ones((2, 2, 2), dtype=dtype)
        arr4d = np.ones((2, 2, 2, 2), dtype=dtype)

        # test invalid shapes
        with pytest.raises(TypeError):
            nanmean_image_data(arr4d)
        with pytest.raises(TypeError):
            nanmean_image_data(arr1d)
        with pytest.raises(TypeError):
            nanmean_image_data(arr2d, arr2d, arr2d)

        # ---
        # input is an array of images
        # ---

        # kept is an empty list
        with pytest.raises(ValueError):
            nanmean_image_data(arr3d, kept=[])

        # input is a 2D array
        data = np.random.randn(2, 2)
        ret = nanmean_image_data(data)
        np.testing.assert_array_equal(data, ret)
        assert ret is not data

        # input is a 3D array
        data = np.array([[[np.nan,       2, np.nan], [     1, 2, -np.inf]],
                         [[     1, -np.inf, np.nan], [np.nan, 3,  np.inf]],
                         [[np.inf,       4, np.nan], [     1, 4,      1]]], dtype=dtype)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Note that mean of -np.inf, np.inf and 1 are np.nan!!!
            expected = np.array([[np.inf, -np.inf, np.nan], [  1, 3,  np.nan]], dtype=dtype)
            np.testing.assert_array_almost_equal(expected, np.nanmean(data, axis=0))
            np.testing.assert_array_almost_equal(expected, nanmean_image_data(data))

            # test nanmean on the sliced array
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3, ...], axis=0),
                                                 nanmean_image_data(data, kept=[0, 1, 2]))
            np.testing.assert_array_almost_equal(np.nanmean(data[1:2, ...], axis=0),
                                                 nanmean_image_data(data, kept=[1]))
            np.testing.assert_array_almost_equal(np.nanmean(data[0:3:2, ...], axis=0),
                                                 nanmean_image_data(data, kept=[0, 2]))

        # ---
        # input are two images
        # ---

        # test two images have different shapes
        with pytest.raises(ValueError):
            nanmean_image_data(arr2d, np.ones((2, 3), dtype=dtype))

        # test two images have different dtype
        with pytest.raises(TypeError):
            nanmean_image_data(
                arr2d, np.ones((2, 3), dtype=np.float64 if dtype == np.float32 else np.float32))

        # input are a list/tuple of two images
        img1 = np.array([[1, 1, 2], [np.inf, np.nan, 0]], dtype=dtype)
        img2 = np.array([[np.nan, 0, 4], [2, np.nan, -np.inf]], dtype=dtype)
        expected = np.array([[1., 0.5, 3], [np.inf, np.nan, -np.inf]])
        np.testing.assert_array_almost_equal(expected, nanmean_image_data(img1, img2))

    def testMovingAverage(self):
        dtype = IMAGE_DTYPE

        arr1d = np.ones(2, dtype=dtype)
        arr2d = np.ones((2, 2), dtype=dtype)
        arr3d = np.ones((2, 2, 2), dtype=dtype)
        arr4d = np.ones((2, 2, 2, 2), dtype=dtype)

        # test invalid input
        with pytest.raises(TypeError):
            movingAvgImageData()
        with pytest.raises(TypeError):
            movingAvgImageData(arr1d, arr1d, 2)
        with pytest.raises(TypeError):
            movingAvgImageData(arr4d, arr4d, 2)

        # count is 0
        with pytest.raises(ValueError):
            movingAvgImageData(arr2d, arr2d, 0)
        with pytest.raises(ValueError):
            movingAvgImageData(arr3d, arr3d, 0)

        # inconsistent shape
        with pytest.raises(TypeError):
            movingAvgImageData(arr2d, arr3d)
        with pytest.raises(ValueError):
            movingAvgImageData(arr2d, np.ones((2, 3), dtype=dtype), 2)
        with pytest.raises(ValueError):
            movingAvgImageData(arr3d, np.ones((2, 3, 2), dtype=dtype), 2)

        # inconsistent dtype
        with pytest.raises(TypeError):
            movingAvgImageData(arr2d, np.ones((2, 2), dtype=np.float64), 2)
        with pytest.raises(TypeError):
            movingAvgImageData(arr3d, np.ones((2, 2, 2), dtype=np.float64), 2)

        # ------------
        # single image
        # ------------

        img1 = np.array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
        img2 = np.array([[2, 3, 4], [4, 5, 6]], dtype=dtype)
        movingAvgImageData(img1, img2, 2)
        ma_gt = np.array([[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]], dtype=dtype)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, 2, 3], [3, 4, 5]],
                          [[1, 2, 3], [3, 4, 5]]], dtype=dtype)
        imgs2 = np.array([[[2, 3, 4], [4, 5, 6]],
                          [[2, 3, 4], [4, 5, 6]]], dtype=dtype)
        movingAvgImageData(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]],
                         [[1.5, 2.5, 3.5], [3.5, 4.5, 5.5]]], dtype=dtype)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testMovingAverageWithNan(self):
        dtype = IMAGE_DTYPE

        # ------------
        # single image
        # ------------

        img1 = np.array([[1, np.nan, 3], [np.nan, 4, 5]], dtype=dtype)
        img2 = np.array([[2,      3, 4], [np.nan, 5, 6]], dtype=dtype)
        movingAvgImageData(img1, img2, 2)
        ma_gt = np.array([[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]], dtype=dtype)

        np.testing.assert_array_equal(ma_gt, img1)

        # ------------
        # train images
        # ------------

        imgs1 = np.array([[[1, np.nan, 3], [np.nan, 4, 5]],
                          [[1,      2, 3], [np.nan, 4, 5]]], dtype=dtype)
        imgs2 = np.array([[[2,      3, 4], [     4, 5, 6]],
                          [[2,      3, 4], [     4, 5, 6]]], dtype=dtype)
        movingAvgImageData(imgs1, imgs2, 2)
        ma_gt = np.array([[[1.5, np.nan, 3.5], [np.nan, 4.5, 5.5]],
                          [[1.5,    2.5, 3.5], [np.nan, 4.5, 5.5]]], dtype=dtype)

        np.testing.assert_array_equal(ma_gt, imgs1)

    def testCorrectImageData(self):
        dtype = IMAGE_DTYPE

        arr1d = np.ones(2, dtype=dtype)
        arr2d = np.ones((2, 2), dtype=dtype)
        arr3d = np.ones((2, 2, 2), dtype=dtype)
        arr4d = np.ones((2, 2, 2, 2), dtype=dtype)

        # test invalid input
        with pytest.raises(TypeError):
            correct_image_data()
        with pytest.raises(TypeError):
            correct_image_data(arr1d, offset=arr1d)
        with pytest.raises(TypeError):
            correct_image_data(arr4d, gain=arr4d)

        # test incorrect shape
        with pytest.raises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), offset=arr2d)
        with pytest.raises(TypeError):
            correct_image_data(np.ones((2, 2, 2)), gain=arr2d)
        with pytest.raises(TypeError):
            correct_image_data(np.ones((2, 2)), offset=arr3d)
        with pytest.raises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=arr3d)
        with pytest.raises(TypeError):
            correct_image_data(np.ones((2, 2)), gain=arr2d, offset=arr3d)

        # test incorrect dtype
        with pytest.raises(TypeError):
            correct_image_data(arr3d, offset=np.ones((2, 2, 2), dtype=np.float64))
        with pytest.raises(TypeError):
            correct_image_data(arr3d, gain=arr3d, offset=np.ones((2, 2, 2), dtype=np.float64))

        # test without gain and offset
        for img in [np.ones([2, 2]), np.ones([2, 2, 2])]:
            img_gt = img.copy()
            correct_image_data(img)
            np.testing.assert_array_equal(img_gt, img)

        # ------------
        # single image
        # ------------

        # offset only
        img = np.array([[1, 2, 3], [3, np.nan, np.nan]], dtype=dtype)
        offset = np.array([[1, 2, 1], [2, np.nan, np.nan]], dtype=dtype)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [1, np.nan, np.nan]], dtype=dtype), img)

        # gain only
        gain = np.array([[1, 2, 1], [2, 2, 1]], dtype=dtype)
        correct_image_data(img, gain=gain)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [2, np.nan, np.nan]], dtype=dtype), img)

        # both gain and offset
        img = np.array([[1, 2, 3], [3, np.nan, np.nan]], dtype=dtype)
        correct_image_data(img, gain=gain, offset=offset)
        np.testing.assert_array_equal(
            np.array([[0, 0, 2], [2, np.nan, np.nan]], dtype=dtype), img)

        # ------------
        # train images
        # ------------

        # offset only
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=dtype)
        offset = np.array([[[1, 2, 1], [3, np.nan, np.nan]],
                           [[2, 1, 2], [2, np.nan, np.nan]]], dtype=dtype)
        correct_image_data(img, offset=offset)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-1, 1, 1], [1, np.nan, np.nan]]],
                                               dtype=dtype), img)

        # including DSSC and intradark
        img = np.array([[[200, 210, 220], [200, 210, 220]],
                        [[200, 220, 0], [200, 220, 0]]], dtype=dtype)
        offset = np.array([[[40, 45, 50], [40, 45, 50]],
                           [[40, 50, 45], [40, 50, 45]]], dtype=dtype)
        correct_image_data(img, offset=offset, intradark=True, detector="DSSC")
        np.testing.assert_array_equal(np.array([[[0, -5, -41], [0, -5, -41]],
                                                [[160, 170, 211], [160, 170, 211]]],
                                               dtype=dtype), img)

        # gain only
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=dtype)
        gain = np.array([[[1, 2, 1], [2, 2, 1]],
                         [[2, 1, 2], [2, 1, 2]]], dtype=dtype)
        correct_image_data(img, gain=gain)
        np.testing.assert_array_equal(np.array([[[1, 4, 3], [6, np.nan, np.nan]],
                                                [[2, 2, 6], [6, np.nan, np.nan]]],
                                               dtype=dtype), img)

        # both gain and offset
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=dtype)
        gain = np.array([[[1, 2, 1], [2, 2, 1]],
                         [[2, 1, 2], [2, 1, 2]]], dtype=dtype)
        offset = np.array([[[1, 2, 1], [3, np.nan, np.nan]],
                           [[2, 1, 2], [2, np.nan, np.nan]]], dtype=dtype)
        correct_image_data(img, gain=gain, offset=offset)
        np.testing.assert_array_equal(np.array([[[0, 0, 2], [0, np.nan, np.nan]],
                                                [[-2, 1, 2], [2, np.nan, np.nan]]],
                                               dtype=dtype), img)


class TestMaskImageData:
    @pytest.mark.parametrize("keep_nan, mt", [(False, 0), (True, np.nan)])
    def testMaskImageData(self, keep_nan, mt):
        dtype = IMAGE_DTYPE

        arr1d = np.ones(2, dtype=dtype)
        arr2d = np.ones((2, 2), dtype=dtype)
        arr3d = np.ones((2, 2, 2), dtype=dtype)
        arr4d = np.ones((2, 2, 2, 2), dtype=dtype)

        # test invalid input
        with pytest.raises(TypeError):
            mask_image_data()
        with pytest.raises(TypeError):
            mask_image_data(arr1d, threshold_mask=(1, 2), keep_nan=keep_nan)
        with pytest.raises(TypeError):
            mask_image_data(arr4d, threshold_mask=(1, 2), keep_nan=keep_nan)

        # test inconsistent shape
        with pytest.raises(TypeError):
            mask_image_data(arr2d, image_mask=arr3d, threshold_mask=(1, 2), keep_nan=keep_nan)
        with pytest.raises(TypeError):
            mask_image_data(arr3d, image_mask=arr2d, threshold_mask=(1, 2), keep_nan=keep_nan)
        with pytest.raises(TypeError):
            mask_image_data(arr3d, image_mask=arr3d, threshold_mask=(1, 2), keep_nan=keep_nan)
        with pytest.raises(ValueError):
            mask_image_data(arr3d, image_mask=np.ones((3, 2), dtype=bool), keep_nan=keep_nan)
        with pytest.raises(ValueError):
            mask_image_data(arr2d, image_mask=np.ones((3, 2), dtype=bool), keep_nan=keep_nan)

        # test inconsistent dtype
        with pytest.raises(TypeError):
            mask_image_data(arr3d, image_mask=np.ones((2, 2), dtype=int), keep_nan=keep_nan)

        # ------------
        # single image
        # ------------

        # raw
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=dtype)
        mask_image_data(img, keep_nan=keep_nan)
        np.testing.assert_array_equal(
            np.array([[1, 2, mt], [3, 4, 5]], dtype=dtype), img)

        # threshold mask
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=dtype)
        mask_image_data(img, threshold_mask=(2, 3), keep_nan=keep_nan)
        np.testing.assert_array_equal(
            np.array([[mt, 2, mt], [3, mt, mt]], dtype=dtype), img)

        # image mask
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=dtype)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        mask_image_data(img, image_mask=img_mask, keep_nan=keep_nan)
        np.testing.assert_array_equal(
            np.array([[mt, mt, mt], [mt, 4, mt]], dtype=dtype), img)

        # both masks
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=dtype)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        mask_image_data(img, image_mask=img_mask, threshold_mask=(2, 3), keep_nan=keep_nan)
        np.testing.assert_array_equal(
            np.array([[mt, mt, mt], [mt, mt, mt]], dtype=dtype), img)

        # ------------
        # train images
        # ------------

        # raw
        img = np.array([[[1, 2, 3], [3, np.nan, 5]],
                        [[1, 2, 3], [3, np.nan, 5]]], dtype=dtype)
        mask_image_data(img, keep_nan=keep_nan)
        np.testing.assert_array_equal(np.array([[[1, 2, 3], [3, mt, 5]],
                                                [[1, 2, 3], [3, mt, 5]]], dtype=dtype), img)

        # threshold mask
        img = np.array([[[1, 2, 3], [3, np.nan, 5]],
                        [[1, 2, 3], [3, np.nan, 5]]], dtype=dtype)
        mask_image_data(img, threshold_mask=(2, 3), keep_nan=keep_nan)
        np.testing.assert_array_equal(np.array([[[mt, 2, 3], [3, mt, mt]],
                                                [[mt, 2, 3], [3, mt, mt]]], dtype=dtype), img)

        # image mask
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 3], [3, np.nan, np.nan]]], dtype=dtype)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        mask_image_data(img, image_mask=img_mask, keep_nan=keep_nan)
        np.testing.assert_array_equal(np.array([[[mt, mt, 3], [mt, mt, mt]],
                                                [[mt, mt, 3], [mt, mt, mt]]], dtype=dtype), img)

        # both masks
        img = np.array([[[1, 2, 3], [3, np.nan, np.nan]],
                        [[1, 2, 6], [3, np.nan, np.nan]]], dtype=dtype)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        mask_image_data(img, image_mask=img_mask, threshold_mask=(2, 4), keep_nan=keep_nan)
        np.testing.assert_array_equal(np.array([[[mt, mt, 3], [mt, mt, mt]],
                                                [[mt, mt, mt], [mt, mt, mt]]], dtype=dtype), img)

    @pytest.mark.parametrize("keep_nan, mt", [(False, 0), (True, np.nan)])
    def testMaskImageDataWithOutput(self, keep_nan, mt):
        dtype = IMAGE_DTYPE

        arr1d = np.ones(2, dtype=dtype)
        arr2d = np.ones((2, 2), dtype=dtype)
        arr3d = np.ones((2, 2, 2), dtype=dtype)

        out = np.ones((2, 2), dtype=bool)

        with pytest.raises(TypeError):
            mask_image_data(arr1d, keep_nan=keep_nan, out=out)
        with pytest.raises(ValueError, match="must be 2D"):
            mask_image_data(arr3d, keep_nan=keep_nan, out=out)
        with pytest.raises(TypeError):
            mask_image_data(arr2d, image_mask=arr3d, keep_nan=keep_nan, out=out)
        with pytest.raises(TypeError):
            mask_image_data(arr2d, image_mask=arr1d, keep_nan=keep_nan, out=out)
        with pytest.raises(ValueError, match="must be bool"):
            mask_image_data(arr2d, image_mask=arr1d, keep_nan=keep_nan, out=out.astype(float))

        # raw
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=dtype)
        out = np.zeros((2, 3), dtype=bool)
        mask_image_data(img, keep_nan=keep_nan, out=out)
        np.testing.assert_array_equal(
            np.array([[False, False, True], [False, False, False]], dtype=bool), out)

        # threshold mask
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=dtype)
        out = np.zeros((2, 3), dtype=bool)
        mask_image_data(img, threshold_mask=(2, 3), keep_nan=keep_nan, out=out)
        np.testing.assert_array_equal(
            np.array([[mt, 2, mt], [3, mt, mt]], dtype=dtype), img)
        np.testing.assert_array_equal(
            np.array([[True, False, True], [False, True, True]], dtype=bool), out)

        # image mask
        img = np.array([[1, np.nan, np.nan], [3, 4, 5]], dtype=dtype)
        img_mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
        out = np.zeros((2, 3), dtype=bool)
        mask_image_data(img, image_mask=img_mask, keep_nan=keep_nan, out=out)
        np.testing.assert_array_equal(np.array([[mt, mt, mt], [mt, 4, mt]], dtype=dtype), img)
        np.testing.assert_array_equal(
            np.array([[True, True, True], [True, False, True]], dtype=bool), out)

        # both masks
        img = np.array([[1, 2, np.nan], [3, 4, 5]], dtype=dtype)
        img_mask = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
        out = np.zeros((2, 3), dtype=bool)
        mask_image_data(img, image_mask=img_mask, threshold_mask=(2, 3), keep_nan=keep_nan, out=out)
        np.testing.assert_array_equal(
            np.array([[mt, 2, mt], [mt, mt, mt]], dtype=dtype), img)
        np.testing.assert_array_equal(
            np.array([[True, False, True], [True, True, True]], dtype=bool), out)
