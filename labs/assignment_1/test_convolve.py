import unittest
import numpy as np
from scipy.signal import correlate2d
from HM1_Convolve import padding, convol_with_Toeplitz_matrix, convolve


class TestPadding(unittest.TestCase):
    def test_zero_padding(self):
        img = np.array([[1, 2], [3, 4]])
        padding_size = 2
        type = "zeroPadding"
        expected_output = np.pad(img, padding_size, "constant", constant_values=0)
        output = padding(img, padding_size, type)
        np.testing.assert_array_equal(output, expected_output)

    def test_replicate_padding(self):
        img = np.array([[1, 2], [3, 4]])
        padding_size = 2
        type = "replicatePadding"
        expected_output = np.pad(img, padding_size, "edge")
        output = padding(img, padding_size, type)
        np.testing.assert_array_equal(output, expected_output)


class TestConvolve(unittest.TestCase):
    def test_convolve_toeplitz(self):
        """
        The size is fixed to 6*6 and 3*3.
        """
        img = np.random.rand(6, 6)
        kernel = np.random.rand(3, 3)
        output = convol_with_Toeplitz_matrix(img, kernel)
        expected_output = correlate2d(img, kernel, mode="same")
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_convolve(self):
        img = np.random.rand(8, 6)
        padding_img = padding(img, 1, "zeroPadding")
        kernel = np.random.rand(3, 3)
        output = convolve(padding_img, kernel)
        expected_output = correlate2d(img, kernel, mode="same")
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_filter(self):
        img = np.random.rand(8, 6)
        padding_img = padding(img, 1, "replicatePadding")
        kernel = np.array(
            [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]
        )
        output = convolve(padding_img, kernel)
        expected_output = correlate2d(padding_img, kernel)[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_sobel(self):
        img = np.random.rand(8, 6)
        padding_img = padding(img, 1, "replicatePadding")
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        output_x = convolve(padding_img, kernel_x)
        output_y = convolve(padding_img, kernel_y)
        expected_output_x = correlate2d(padding_img, kernel_x)[2:-2, 2:-2]
        expected_output_y = correlate2d(padding_img, kernel_y)[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(output_x, expected_output_x)
        np.testing.assert_array_almost_equal(output_y, expected_output_y)


if __name__ == "__main__":
    unittest.main()
