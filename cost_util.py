from enum import Enum
import cv2
import numpy as np


class CostType(Enum):
    L1 = 1
    L2 = 2


class MatchType(Enum):
    PIXEL_WISE = 1
    PIXEL_WINDOWS = 2


def l1(x, y):
    return abs(x - y)


def l2(x, y):
    return (x - y) * (x - y)


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    return numerator / denominator


def cost(x, y, cost_type):
    if cost_type == 1:
        return l1(x, y)
    elif cost_type == 2:
        return l2(x, y)
    elif cost_type == 3:
        return cosine_similarity(x, y)


def prepare_image(left_img, right_img):
    left = cv2.imread(left_img, 0).astype(np.float32)
    right = cv2.imread(right_img, 0).astype(np.float32)

    height, width = left.shape[:2]

    return left, right, height, width


def get_max_value(cost_type, matching_type):
    if matching_type == MatchType.PIXEL_WINDOWS:
        if cost_type == CostType.L1:
            return 255 * 9
        else:
            return 255 * 255
    else:
        if cost_type == CostType.L1:
            return 255
        else:
            return 255 * 255
