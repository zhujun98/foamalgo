import pytest

import numpy as np
from scipy.signal import find_peaks

from pyfoamalgo import AzimuthalIntegrator, ConcentricRingsFinder

_AVAILABLE_DTYPES = [np.float64, np.float32, np.uint16, np.int16]


def create_image(w, h, cx, cy, *, aspect_ratio=1., lw=2, radius=None, dtype=np.float32):
    if cx is None:
        cx = int(w / 2)
    if cy is None:
        cy = int(h / 2)

    img = np.zeros((w, h), dtype=dtype)

    if radius is None:
        radius = [20, 100, 130, 200, 300]

    for r in radius:
        for theta in np.linspace(0, 360, 10000):
            y = cy + aspect_ratio * r * np.cos(theta) + (2 * np.random.random_sample() - 1.)
            x = cx + r * np.sin(theta) + (2 * np.random.random_sample() - 1.)
            img[int(y-lw/2):int(y+lw/2), int(x-lw/2):int(x+lw/2)] = 1

    return img


def maybe_mask_image(img):
    h, w = img.shape
    if isinstance(img.dtype, np.floating):
        img[:, 100:110] = np.nan
        img[int(h/2):int(h/2) + 10, :] = np.nan


class TestAzimuthalIntegrator:
    @classmethod
    def setup_class(cls):
        h, w = 640, 480
        cy, cx = 400, 320
        pixel1, pixel2 = 2e-4, 1e-4
        poni1, poni2 = cy * pixel1, cx * pixel2
        wavelength = 1e-10
        distance = 1.

        ratio = pixel2 / pixel1
        cls._img1 = create_image(w, h, cx, cy, dtype=np.float64, aspect_ratio=ratio)
        cls._img2 = create_image(w, h, cx, cy, dtype=np.float64, aspect_ratio=ratio)

        cls._integrator = AzimuthalIntegrator(
            dist=distance, poni1=poni1, poni2=poni2, pixel1=pixel1, pixel2=pixel2,
            wavelength=wavelength)

    @pytest.mark.parametrize("dtype", _AVAILABLE_DTYPES)
    def test_integrate1d(self, dtype):
        integrator = self._integrator
        img = self._img1.astype(dtype)
        maybe_mask_image(img)

        # tet corner cases
        q0, s0 = integrator.integrate1d(img, npt=0)
        q1, s1 = integrator.integrate1d(img, npt=1)
        assert q0 == q1
        assert s0 == s1

        # test threshold
        q10, s10 = integrator.integrate1d(img, npt=10, min_count=img.size)
        assert not np.any(s10)

        # test correctness
        q512, s512 = integrator.integrate1d(img, npt=512)
        assert 512 == len(q512)
        assert 512 == len(s512)
        assert 0.539348531 == pytest.approx(1e-10 * q512[-1], abs=1e-6)
        peaks, _ = find_peaks(s512)
        np.testing.assert_array_equal([11,  59,  77, 119, 178], peaks)
        for peak in peaks:
            assert 0.9 <= s512[peak] <= 1.0

    @pytest.mark.parametrize("dtype", _AVAILABLE_DTYPES)
    def test_integrate1d_array(self, dtype):
        img1 = self._img1.astype(dtype)
        img2 = self._img2.astype(dtype)
        img_a = np.array([img1, img2])

        integrator = self._integrator

        q1, s1 = integrator.integrate1d(img1, npt=512)
        q2, s2 = integrator.integrate1d(img2, npt=512)
        q_a, s_a = integrator.integrate1d(img_a, npt=512)

        np.testing.assert_array_equal(q1, q_a)
        np.testing.assert_array_equal(q2, q_a)
        np.testing.assert_array_equal(s1, s_a[0])
        np.testing.assert_array_equal(s2, s_a[1])


class TestConcentricRingsFinder:
    @classmethod
    def setup_class(cls):
        h, w = 640, 480
        cls._cy, cls._cx = 400, 320
        pixel_y, pixel_x = 2e-4, 1e-4

        ratio = pixel_x / pixel_y
        cls._img = create_image(w, h, cls._cx, cls._cy, dtype=np.float64, aspect_ratio=ratio)
        cls._finder = ConcentricRingsFinder(pixel_x, pixel_y)

    @pytest.mark.parametrize("dtype", _AVAILABLE_DTYPES)
    def test_ring_detection(self, dtype):
        img = self._img.astype(dtype)
        maybe_mask_image(img)

        cy0, cx0 = self._cy + 8, self._cx - 8
        cx_opt, cy_opt = self._finder.search(img, cx0, cy0, min_count=1)
        assert abs(cx_opt - self._cx) <= 1
        assert abs(cy_opt - self._cy) <= 1
