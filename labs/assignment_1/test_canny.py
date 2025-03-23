import unittest
import numpy as np
from scipy.signal import correlate2d
from scipy.interpolate import RegularGridInterpolator
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from HM1_Canny import (
    compute_gradient_magnitude_direction,
    non_maximal_suppressor,
    bilinear_interpolation,
)


class TestNMS(unittest.TestCase):
    def test_bilinear_inter(self):
        grad_mag = np.random.rand(6, 6)
        x = np.arange(0.5, 1.6, 0.3)
        y = np.arange(0.5, 1.6, 0.3)
        xx, yy = np.meshgrid(x, y)
        target_grad = bilinear_interpolation(xx, yy, grad_mag)

        interpolator = RegularGridInterpolator(
            (np.arange(6), np.arange(6)),
            grad_mag,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        points = np.array([yy.flatten(), xx.flatten()]).T
        expected = interpolator(points).reshape(xx.shape)

        np.testing.assert_allclose(target_grad, expected)


if __name__ == "__main__":
    unittest.main()
