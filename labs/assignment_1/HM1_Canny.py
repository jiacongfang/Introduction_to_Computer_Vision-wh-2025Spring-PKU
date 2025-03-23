import os
import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, padding
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
    The function you need to implement for Q2 a).
    Inputs:
        x_grad: array(float)
        y_grad: array(float)
    Outputs:
        magnitude_grad: array(float)
        direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad


def bilinear_interpolation(x, y, grad_mag):
    """
    Bilinear interpolation to get gradient of target postion (x, y)
    Inputs:
        x: array(float), target position_x
        y: array(float), target position_y
        grad_mag: array(float), magnitude_grad

    Outputs:
        target_grad: array(float), the estimated gradient of target position
    """
    H, W = grad_mag.shape

    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)

    f00, f01, f10, f11 = (
        grad_mag[y0, x0],
        grad_mag[y0, x1],
        grad_mag[y1, x0],
        grad_mag[y1, x1],
    )

    mid_0 = (x1 - x) * f00 + (x - x0) * f01
    mid_1 = (x1 - x) * f10 + (x - x0) * f11

    target_grad = (y1 - y) * mid_0 + (y - y0) * mid_1
    return target_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
    The function you need to implement for Q2 b).
    **Implement the full version with bilinear interpolation.**
    Inputs:
        grad_mag: array(float)
        grad_dir: array(float)
    Outputs:
        output: array(float)
    """
    H, W = grad_mag.shape

    # get the coordinates of the pixels along and against the gradient direction
    indice_x, indice_y = np.meshgrid(np.arange(W), np.arange(H))

    along_grad_x = np.maximum(0, indice_x - np.cos(grad_dir) * grad_mag)
    along_grad_y = np.maximum(0, indice_y - np.sin(grad_dir) * grad_mag)

    against_grad_x = np.minimum(indice_x + np.cos(grad_dir) * grad_mag, W - 1)
    against_grad_y = np.minimum(indice_y + np.sin(grad_dir) * grad_mag, H - 1)

    along_grad = bilinear_interpolation(along_grad_x, along_grad_y, grad_mag)
    against_grad = bilinear_interpolation(against_grad_x, against_grad_y, grad_mag)

    mask = (grad_mag > along_grad) & (grad_mag > against_grad)

    output = np.zeros_like(grad_mag)
    output[mask] = grad_mag[mask]

    return output


def hysteresis_thresholding(img):
    """
    The function you need to implement for Q2 c).
    Inputs:
        img: array(float)
    Outputs:
        output: array(float)
    """

    # you can adjust the parameters to fit your own implementation
    high_ratio = 0.42
    low_ratio = 0.2

    output = np.zeros_like(img).astype(np.uint8)

    mask_high = img >= high_ratio
    unknown = (img >= low_ratio) & (img < high_ratio)

    output[mask_high] = 1

    # store the previous output to check if the output is still changing
    prev_output = np.zeros_like(img)
    while not np.array_equal(output, prev_output):
        prev_output = output.copy()
        dilated = np.zeros_like(output)
        # Loop through the eight neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                padding_output = padding(output, 1, "zeroPadding")
                padding_shifted = np.roll(padding_output, (dy, dx), axis=(0, 1))
                shifted = padding_shifted[1:-1, 1:-1].astype(np.uint8)
                dilated = np.logical_or(dilated, shifted)

        new_edges = np.logical_and(unknown, dilated)
        output = np.logical_or(output, new_edges)

    return output.astype(float)


if __name__ == "__main__":
    # Load the input images
    input_img = read_img("Lenna.png") / 255

    if not os.path.exists("result"):
        os.makedirs("result")

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad
    )

    write_img("result/HM1_magnitude_grad.png", magnitude_grad * 255)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    write_img("result/HM1_NMS_result.png", NMS_output * 255)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img * 255)
