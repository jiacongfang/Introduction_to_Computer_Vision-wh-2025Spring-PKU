import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
    The function you need to implement for Q1 a).
    Inputs:
        img: array(float)
        padding_size: int
        type: str, zeroPadding/replicatePadding
    Outputs:
        padding_img: array(float)
    """

    if type == "zeroPadding":
        padding_img = np.zeros(
            (img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size)
        )
        padding_img[
            padding_size : padding_size + img.shape[0],
            padding_size : padding_size + img.shape[1],
        ] = img
        return padding_img

    elif type == "replicatePadding":
        padding_img = np.zeros(
            (img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size)
        )
        padding_img[
            padding_size : padding_size + img.shape[0],
            padding_size : padding_size + img.shape[1],
        ] = img

        top, down = padding_size, padding_size + img.shape[0]
        left, right = padding_size, padding_size + img.shape[1]
        padding_img[0:top, left:right] = img[0, :]
        padding_img[down:, left:right] = img[-1, :]
        padding_img[:, 0:left] = padding_img[:, left].reshape(-1, 1)
        padding_img[:, right:] = padding_img[:, right - 1].reshape(-1, 1)

        # padding the corner
        padding_img[0:top, 0:left] = img[0, 0]
        padding_img[down:, 0:left] = img[-1, 0]
        padding_img[0:top, right:] = img[0, -1]
        padding_img[down:, right:] = img[-1, -1]

        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
    The function you need to implement for Q1 b).
    Inputs:
        img: array(float) 6*6
        kernel: array(float) 3*3
    Outputs:
        output: array(float) 6*6

    Note: Teplitz matrix 36 * 64
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")

    # build the Toeplitz matrix and compute convolution
    toeplitz_matrix = np.zeros((36, 64))

    indices_x = np.arange(36)
    indices_y = np.arange(64).reshape(8, 8)

    # indices for kernel a_11, a_12, ... , a_33
    indices_y1 = indices_y[0:6, 0:6].flatten()
    indices_y2 = indices_y[0:6, 1:7].flatten()
    indices_y3 = indices_y[0:6, 2:8].flatten()
    indices_y4 = indices_y[1:7, 0:6].flatten()
    indices_y5 = indices_y[1:7, 1:7].flatten()
    indices_y6 = indices_y[1:7, 2:8].flatten()
    indices_y7 = indices_y[2:8, 0:6].flatten()
    indices_y8 = indices_y[2:8, 1:7].flatten()
    indices_y9 = indices_y[2:8, 2:8].flatten()

    # build the Toeplitz matrix
    toeplitz_matrix[indices_x, indices_y1] = kernel[0, 0]
    toeplitz_matrix[indices_x, indices_y2] = kernel[0, 1]
    toeplitz_matrix[indices_x, indices_y3] = kernel[0, 2]
    toeplitz_matrix[indices_x, indices_y4] = kernel[1, 0]
    toeplitz_matrix[indices_x, indices_y5] = kernel[1, 1]
    toeplitz_matrix[indices_x, indices_y6] = kernel[1, 2]
    toeplitz_matrix[indices_x, indices_y7] = kernel[2, 0]
    toeplitz_matrix[indices_x, indices_y8] = kernel[2, 1]
    toeplitz_matrix[indices_x, indices_y9] = kernel[2, 2]

    output = (toeplitz_matrix @ padding_img.flatten()).reshape(6, 6)
    return output


def convolve(img, kernel):
    """
    The function you need to implement for Q1 c).
    Inputs:
        img: array(float) M*N
        kernel: array(float) k*k
    Outputs:
        output: array(float)
    """
    # build the sliding-window convolution here
    M, N = img.shape
    k = kernel.shape[0]
    output_h, output_w = M - k + 1, N - k + 1

    i, j = np.meshgrid(np.arange(output_h), np.arange(output_w), indexing="ij")

    dx, dy = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")

    # Expand the dimensions for broadcasting
    i_expanded, j_expanded = (
        i[..., np.newaxis, np.newaxis],
        j[..., np.newaxis, np.newaxis],
    )

    # Auto Broadcasting
    windows = img[i_expanded + dx, j_expanded + dy]

    output = (windows.reshape(-1, k * k)) @ kernel.flatten()

    return output.reshape(output_h, output_w)


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array(
        [[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]
    )
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":
    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
