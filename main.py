import cv2

import LSD_based_lines_and_cross_finding as LSD

if __name__ == '__main__':
    img = cv2.imread('test-image/2.png')
    cross_list = LSD.lines_and_cross_finding(img)
    print(cross_list)
    pass
