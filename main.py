import cv2 as cv

img_org = cv.imread('test-image/screenshot.png', 0)


def show(img):
    cv.imshow('img', img)
    cv.waitKey(0)


def 高斯模糊(img=img_org):
    """
    高斯模糊可以减少搜索到的边缘数量
    :param img:
    :return:
    """
    img_Blur = cv.GaussianBlur(img, (5, 5), 0)
    show(img_Blur)
    return img_Blur


def 边缘检测(img=img_org):
    img_canny = cv.Canny(img, 100, 200)
    show(img_canny)
    return img_canny


if __name__ == '__main__':
    边缘检测()
