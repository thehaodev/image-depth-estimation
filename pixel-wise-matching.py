from enum import Enum
import cv2
import numpy as np
import cost_util

left_img_path = 'tsukuba/left.png'
right_img_path = 'tsukuba/right.png'


class CostType(Enum):
    L1 = 1
    L2 = 2


def prepare_image(left_img, right_img, cost_type):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)

    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    if cost_type == CostType.L1:
        max_value = 255
    else:
        max_value = 255 * 255

    return left, right, height, width, depth, scale, max_value


def pixel_wise_matching(left_img, right_img, cost_type, save_result=True):
    disparity_range = 16
    left, right, height, width, depth, scale, max_value = prepare_image(left_img, right_img, cost_type)

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value
            for j in range(disparity_range):
                cost = max_value if (x - j) < 0 else cost_util.cost(int(left[y, x]),
                                                                    int(right[y, x - j]),
                                                                    cost_type)

                if cost < cost_min:
                    cost_min = cost
                    disparity = j

            depth[y, x] = disparity * scale

    if save_result:
        print('Saving result... ')
        cv2.imwrite(f'pixel_wise_{cost_type}.png', depth)
        cv2.imwrite(f'pixel_wise_{cost_type}_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done')

    return depth


pixel_wise_matching(left_img_path, right_img_path, save_result=True, cost_type=CostType.L2)
