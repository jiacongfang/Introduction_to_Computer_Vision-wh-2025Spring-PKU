import unittest
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    from HM1_Canny import bilinear_interpolation

    HAS_BILINEAR_INTERPOLATION = True
except ImportError:
    HAS_BILINEAR_INTERPOLATION = False


class TestNMS(unittest.TestCase):
    @unittest.skipIf(
        not HAS_BILINEAR_INTERPOLATION,
        "bilinear_interpolation not implemented in HM1_Canny",
    )
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
