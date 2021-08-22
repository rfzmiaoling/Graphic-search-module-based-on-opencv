import cv2


def LSD(img_origin):
    """
    通过LSD算法寻找图片中的直线，返回一个直线起点和终点的列表
    :param img_origin: 需要搜索的原图
    :return: 直线起点和终点的列表
    """
    img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD, _scale=0.8, _sigma_scale=0.6, _quant=2.0,
                                        _ang_th=22.5, _log_eps=0, _density_th=0.7, _n_bins=1024)
    lines = lsd.detect(img)
    line_list = []
    for line in lines[0]:
        x0 = int(round(line[0][0]))
        y0 = int(round(line[0][1]))
        x1 = int(round(line[0][2]))
        y1 = int(round(line[0][3]))
        len_line = (abs(y1 - y0) ** 0.5 + abs(x1 - x0) ** 0.5) ** 2
        if len_line > 20:
            x0, y0, x1, y1 = extend_line(x0, y0, x1, y1)
            cv2.line(img_origin, (x0, y0), (x1, y1), (0, 255, 0), 1, cv2.LINE_AA)
            line_list.append((x0, y0, x1, y1))
    return line_list


def extend_line(x1, y1, x2, y2, extend_length=3):
    """
    由于LSD算法找到的直线通常较短，为了产生交点需要延长直线。
    :param x1: 
    :param y1: 
    :param x2: 
    :param y2: 
    :param extend_length: 延长的长度
    :return: 
    """
    if y1 == y2:
        return x1 - 8 * extend_length, y1, x2 + 8 * extend_length, y2
    elif x1 == x2:
        return x1, y1 - 8 * extend_length, x2, y2 + 8 * extend_length
    elif y2 > y1:
        k = (y2 - y1) / (x2 - x1)
        y1 = y1 - extend_length
        y2 = y2 + extend_length
        x1 = x1 - k * extend_length
        x2 = x2 + k * extend_length
        return int(x1), int(y1), int(x2), int(y2)
    elif y2 < y1:
        k = (y1 - y2) / (x1 - x2)
        y1 = y1 + extend_length
        y2 = y2 - extend_length
        x1 = x1 + k * extend_length
        x2 = x2 - k * extend_length
        return int(x1), int(y1), int(x2), int(y2)


def cross_point(line1, line2):
    """
    接收两条直线的起点和终点，返回是否有交叉点，如果有则返回交叉点坐标
    :param line1:第一条直线的起点和终点坐标
    :param line2:第二条直线的起点和终点坐标
    :return:point_is_exist为true时返回交叉点坐标
    """
    global b1
    point_is_exist = False
    x = 0
    y = 0
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            point_is_exist = True
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        point_is_exist = True
    if max(x1, x2) > x > min(x1, x2) and max(x3, x4) > x > min(x3, x4):
        pass
    elif x1 == x2 and max(y1, y2) > y > min(y1, y2) and max(x3, x4) > x > min(x3, x4):
        pass
    elif y1 == y2 and max(x1, x2) > x > min(x1, x2) and max(y3, y4) > y > min(y3, y4):
        pass
    else:
        point_is_exist = False
    return point_is_exist, [x, y]


def cross_points(img_origin, line_list):
    """
    接收要寻找交叉点的原图和直线列表，返回并在图中画出所有交叉点
    :param cross_list: 返回的交叉点列表
    :param img_origin: 用于绘制的原图
    :param line_list: 直线列表
    :return: 返回交叉点列表
    """
    cross_list = []
    for x1, y1, x2, y2 in line_list:
        for x3, y3, x4, y4 in line_list:
            point_is_exist, [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
            if point_is_exist:
                cv2.circle(img_origin, (int(x), int(y)), 2, (0, 0, 255), 2)
                cross_list.append((int(x), int(y)))
    return cross_list


def draw_result(img_origin):
    """
    原图上绘制直线和交叉点
    :param img_origin: 用于绘制的原图
    :return:
    """
    cv2.imwrite('result_img/result.png', img_origin)
    cv2.imshow("LSD", img_origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def lines_and_cross_finding(img_origin):
    """
    在一张图片上寻找直线并寻找直线的交叉点
    :param img_origin: 用于绘制的原图
    :return:
    """
    line_list = LSD(img_origin)
    cross_list = cross_points(img_origin, line_list)
    draw_result(img_origin)
    return cross_list


if __name__ == '__main__':
    img = cv2.imread('test-image/2.png')
    cross_list = lines_and_cross_finding(img)
    print(cross_list)
