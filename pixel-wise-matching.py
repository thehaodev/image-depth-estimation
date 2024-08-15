import cv2
import cost_util
import numpy as np
import time

left_img_path = 'tsukuba/left.png'
right_img_path = 'tsukuba/right.png'


def pixel_wise_matching(left_img, right_img, cost_type, save_result=True):
    scale = 16
    disparity_range = 16
    match_type = cost_util.MatchType.PIXEL_WISE
    left, right, height, width = cost_util.prepare_image(left_img, right_img)
    max_value = cost_util.get_max_value(cost_type, match_type)

    costs = np.full((height, width, disparity_range), max_value, dtype=np.float32)
    for j in range(disparity_range):
        left_d = left[:, j:width]
        right_d = right[:, 0:width - j]
        costs[:, j:width, j] = cost_util.l1(left_d, right_d)

    min_cost_indices = np.argmin(costs, axis=2)

    depth = min_cost_indices * scale
    depth = depth.astype(np.uint8)

    if save_result:
        print('Saving result... ')
        cv2.imwrite('pixel_wise.png', depth)
        cv2.imwrite('pixel_wise_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done')

    print(depth)


start = time.time()
pixel_wise_matching(left_img_path, right_img_path, save_result=True, cost_type=cost_util.CostType.L1)
end = time.time()
print(end - start)
