def l1(x, y):
    return abs(x - y)


def l2(x, y):
    return (x-y) * (x-y)


def cost(x, y, cost_type):
    if cost_type == 1:
        return l1(x, y)
    else:
        return l2(x, y)
