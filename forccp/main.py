import math

import numpy as np
import cv2
import csv
import json

color = {
    1: (45, 35, 192),
    2: (192, 51, 35),
    3: (35, 79, 192),
    4: (35, 192, 35)
}


def read_data(csvfile):
    """read datas from file using csv format"""
    data = []
    with open(csvfile) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row[0] == 'name':
                continue
            name = row[0]
            group_label = int(row[1])
            number = int(row[2])
            target = row[3:]
            if '' in target:
                target.remove('')
            target = [int(t) for t in target]
            data.append([group_label, number, name, target])
        data.sort(key=takegroup)
    return data


def takegroup(data):
    return data[0]


def draw_line(img, source, target, color):
    """draw lines with arrow"""
    tipLength = 0.02
    cv2.arrowedLine(img, source, target, color, tipLength=tipLength, thickness=2, line_type=cv2.LINE_AA)


def draw_circles(img, data, color):
    """draw circles"""
    for d in data:
        cv2.circle(img, (d[1], d[2]), 50, color[d[3]], -1)
        cv2.circle(img, (d[1], d[2]), 48, (255, 255, 255), -1)
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        cv2.putText(img, d[0], (d[1] - 4, d[2] - 4), font, 3.5, color[d[3]], 2)


def divide_eclipse(centerx, centery, a, b, num):
    """divide eclipse into several parts according to the number of elements in group"""
    dangle = 2 * np.pi / num
    centers_circle = []
    for i in range(0, num):
        centers_circle.append([round(a * np.cos(dangle * i) + centerx), round(b * np.sin(dangle * i) + centery)])
    return centers_circle


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


# read data
file = "./data.csv"
data = read_data(file)

# prepare draw background
img = np.zeros((640, 1080, 3), np.uint8)
img[:][:][:] = 255
h, w, c = img.shape

# get the number of elements in each group
group_number = max([d[0] for d in data])
group_label = [i for i in range(1, group_number + 1)]
group = [[] for i in range(0, group_number)]
for i in range(0, len(data)):
    for label in group_label:
        if data[i][0] == label:
            group[label - 1].append(i + 1)

# divide background into several parts

if group_number == 1:
    print(group_number)
elif group_number == 2:
    print(group_number)
elif group_number == 3:
    print(group_number)
elif group_number == 4:
    group_eclipse = [[w / 4, h / 4], [3 * w / 4, h / 4], [w / 4, 3 * h / 4], [3 * w / 4, 3 * h / 4]]
    center_circles = []
    for i in range(0, group_number):
        center_circle = divide_eclipse(group_eclipse[i][0], group_eclipse[i][1], w / 4 - 50, h / 4 - 50, len(group[i]))
        center_circles.append(center_circle)

else:
    print(group_number)

k = 0
for i in range(0, len(center_circles)):
    for j in range(0, len(center_circles[i])):
        data[k].append(center_circles[i][j])
        k = k + 1
dict = {}
for d in data:
    dict.update({d[1]: [d[0], d[2], d[3], d[4]]})
print(dict.values())
for key in dict:
    value = dict.get(key)
    label = value[0]
    name = value[1]
    source_center = value[3]

    for target in value[2]:
        target_value = dict.get(target)
        target_center = target_value[3]
        end_point_of_line = line_intersect_circle(p=(target_center[0], target_center[1], 20),
                                                  lsp=(target_center[0], target_center[1]),
                                                  esp=(source_center[0], source_center[1]))
        end_point_of_line = [round(point) for point in end_point_of_line[0]]
        draw_line(img, source_center, end_point_of_line, color.get(label))

    cv2.circle(img, source_center, 22, color.get(label), -1, cv2.LINE_AA)
    cv2.circle(img, source_center, 18, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, name, (source_center[0] - 20, source_center[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color.get(label), 2)

cv2.imshow("cimg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
a = cv2.imwrite("./picture.jpg", img)
