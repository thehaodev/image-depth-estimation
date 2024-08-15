import numpy as np
import cv2
import cost_util


def window_based_matching(left_img, right_img, disparity_range, cost_type, kernel_size=5):
    matching_type = cost_util.MatchType.PIXEL_WINDOWS
    kernel_half = int((kernel_size - 1) / 2)
    scale = 3

    left, right, height, width = cost_util.prepare_image(left_img, right_img)
    max_value = cost_util.get_max_value(cost_type, matching_type)

    costs = np.full((height, width, disparity_range), max_value, dtype=np.float32)
    for j in range(disparity_range):
        left_j = left[:, j:width]
        right_j = right[:, 0:width-j]

        left_d = left[kernel_half:height - kernel_half, kernel_half + j:width - kernel_half]
        right_d = right[kernel_half:height - kernel_half, kernel_half:width - kernel_half - j]

        cost = np.abs(left_j - right_j)
        cost_d = np.abs(left_d - right_d)

        costs[kernel_half:height-kernel_half, kernel_half+j:width-kernel_half, j] = convolution(kernel_half,
                                                                                                cost_d,
                                                                                                cost)

    min_cost_indices = np.argmin(costs, axis=2)
    depth = min_cost_indices * scale
    depth = depth.astype(np.uint8)

    cv2.imwrite('window_based.png', depth)
    cv2.imwrite('window_based_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
    print("Done")


def convolution(kernel_half, image, origin_image):
    height, width = image.shape
    temp_matrix = np.zeros((height, width))

    for y in range(0, kernel_half + 1):
        for x in range(0, kernel_half + 1):
            temp_matrix += np.abs(origin_image[y: height + y, x: width + x])

    return temp_matrix


left_i = 'Aloe/Aloe_left_1.png'
right_i = 'Aloe/Aloe_right_1.png'

window_based_matching(left_i, right_i, 64, kernel_size=3, cost_type=cost_util.CostType.L1)
