import numpy as np

from LSD_based_lines_and_cross_finding import *

黑色 = (0, 0, 0)
白色 = (255, 255, 255)
红色 = (0, 0, 255)
img = cv2.imread('test-image/2.png')


def show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return


def rectangle(img_origin):
    line_list = LSD(img_origin, entend=True)
    img = np.zeros((720, 1280, 3), np.uint8)
    for line in line_list:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), 白色, 1)
    show(img)


if __name__ == '__main__':
    rectangle(img)
    pass
