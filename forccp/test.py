import math

import cv2
import numpy as np



def create_csys():
    # 创建一个500*500的白色背景图片
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    h, w, c = img.shape
    # 设置坐标轴颜色
    color = (0, 0, 0)
    # 坐标轴x的起始坐标
    piontx1 = (10, int(h / 2))
    pointx2 = (w - 10, int(h / 2))
    # x轴绘制
    cv2.arrowedLine(img, piontx1, pointx2, color)
    # 坐标轴y的起始坐标
    pionty1 = (int(w / 2), h - 10)
    pointy2 = (int(w / 2), 10)
    # y轴绘制
    cv2.arrowedLine(img, pionty1, pointy2, color)
    cv2.imshow("csys img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def divide_eclipse(centerx, centery, a, b, num):
    dangle = 2 * np.pi / num
    centers_circle = []
    for i in range(0, num):
        centers_circle.append([round(a * np.cos(dangle * i) + centerx), round(b * np.sin(dangle * i) + centery)])
    return centers_circle


# 计算圆 与 直接相交的点
def line_intersect_circle(p, lsp, esp):
    # p is the circle parameter, lsp and esp is the two end of the line
    x0, y0, r0 = p
    x1, y1 = lsp
    x2, y2 = esp
    if r0 == 0:
        return [[x1, y1]]
    if x1 == x2:
        if abs(r0) >= abs(x1 - x0):
            p1 = x1, round(y0 - math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
            p2 = x1, round(y0 + math.sqrt(r0 ** 2 - (x1 - x0) ** 2), 5)
            inp = [p1, p2]
            # select the points lie on the line segment
            inp = [p for p in inp if p[0] >= min(x1, x2) and p[0] <= max(x1, x2)]
        else:
            inp = []
    else:
        k = (y1 - y2) / (x1 - x2)
        b0 = y1 - k * x1
        a = k ** 2 + 1
        b = 2 * k * (b0 - y0) - 2 * x0
        c = (b0 - y0) ** 2 + x0 ** 2 - r0 ** 2
        delta = b ** 2 - 4 * a * c
        if delta >= 0:
            p1x = round((-b - math.sqrt(delta)) / (2 * a), 5)
            p2x = round((-b + math.sqrt(delta)) / (2 * a), 5)
            p1y = round(k * p1x + b0, 5)
            p2y = round(k * p2x + b0, 5)
            inp = [[p1x, p1y], [p2x, p2y]]
            # select the points lie on the line segment
            inp = [p for p in inp if min(x1, x2) <= p[0] <= max(x1, x2)]
        else:
            inp = []

    return inp if inp != [] else [[x1, y1]]


if __name__ == "__main__":
    # create_csys()
    img = np.ones((640, 1080, 3), dtype=np.uint8) * 255
    centers_circle = divide_eclipse(270, 160, 220, 110, 8)
    h, w, c = img.shape
    print(centers_circle)
    for i in range(0, len(centers_circle)):
        cv2.circle(img, centers_circle[i], 20, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, centers_circle[i], 18, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, 'QWE', (centers_circle[i][0] - 20, centers_circle[i][1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1)

    cv2.imshow("csys img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    end_point_of_line = line_intersect_circle((426, 238, 20), (426, 238), (114, 82))
    end_point_of_line = [round(point) for point in end_point_of_line[0]]
    print(end_point_of_line)