import numpy as np
from utils import read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x, Sobel_filter_y, padding


def corner_response_function(input_img, window_size, alpha, threshold):
    """
    The function you need to implement for Q3.
    Inputs:
        input_img: array(float)
        window_size: int
        alpha: float
        threshold: float
    Outputs:
        corner_list: array
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.
    x_grad = Sobel_filter_x(input_img)
    y_grad = Sobel_filter_y(input_img)

    x_grad_square, y_grad_square, xy_grad = x_grad**2, y_grad**2, x_grad * y_grad

    rec_kernel = np.ones((window_size, window_size))
    g_xx = convolve(x_grad_square, rec_kernel)
    g_yy = convolve(y_grad_square, rec_kernel)
    g_xy = convolve(xy_grad, rec_kernel)

    theta = (
        g_xx * g_yy - g_xy**2 - alpha * (g_xx + g_yy) ** 2
    )  # corner response function
    mask = theta > threshold

    # Get row and col index of corners
    index = np.argwhere(mask)
    corner_list = np.concatenate((index, theta[mask].reshape(-1, 1)), axis=1)
    return corner_list  # array, each row contains information about one corner, namely (index of row, index of col, theta)


if __name__ == "__main__":
    # Load the input images
    input_img = read_img("hand_writting.png") / 255.0

    # you can adjust the parameters to fit your own implementation
    window_size = 5
    alpha = 0.04
    threshold = 21

    corner_list = corner_response_function(input_img, window_size, alpha, threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = []
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis:
                break
        else:
            NML_selected.append(i[:-1])

    # save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
