import os,sys
import cv2
import numpy as np
from PIL import Image
import pytesseract
import math

def cv2PIL(cv2_image):
    PIL_image = Image.fromarray(cv2_image)
    return PIL_image

# 使用波峰波谷方法进行切割
def extract_peek_ranges_from_array(array_vals, minimun_val=20000, minimun_range=10):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append([start_i, end_i])
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

# 按照中数进行补充切割
def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges

# 按照众数进行补充切割
def mode_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    counts = np.bincount(widthes)
    mode = np.argmax(counts)

    # 当前部分是否被合并
    merged = False
    for i, peek_range in enumerate(peek_ranges):
        if merged:
            merged = False
            continue
        if widthes[i] in range(int(mode * 0.75), int(mode * 1.25)):
            new_peek_ranges.append(peek_range)
        else:
            # 处理合并的情况
            if widthes[i] < 0.75:
                if widthes[i + 1] < mode * 0.5:
                    new_peek_ranges.append((peek_ranges[i][0], peek_ranges[i + 1][1]))
                    merged = True
                else:
                    new_peek_ranges.append(peek_range)
            if widthes[i] >= 1.25:
                new_peek_ranges.append((peek_range[0], int((peek_range[0] + peek_range[1]) / 2)))
                new_peek_ranges.append((int((peek_range[0] + peek_range[1]) / 2), peek_range[1]))
    return new_peek_ranges
    pass

#将整个票据分割为各个部分
def segmentPJ(img, file_path_box_dir):
    # 及其耗时的图片滤波（使用聚类颜色种类）
    # img = cv2.pyrMeanShiftFiltering(img, 25, 10)

    if not os.path.exists(file_path_box_dir):
        os.makedirs(file_path_box_dir)
    heigh , width = img.shape[: 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ = gray.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(gray, 40, 100)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel2)

    _, cnts, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = [(255, 0 ,0), (0, 255, 0), (0, 0, 255), (127,0,127)]
    i = 0
    needToSeg = []
    maxAreaCNT = cnts[0]
    for contour in cnts:
        area = cv2.contourArea(contour)
        if area < heigh/20 * width/40:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 2 * area:
            continue
        if area > cv2.contourArea(maxAreaCNT):
            maxAreaCNT = contour
        gray = cv2.drawContours(gray, contour, -1, color[i%4], -1)
        needToSeg.append((x, y, w, h))
        i+=1
    #不需要分割
    gray = cv2.drawContours(gray, maxAreaCNT, -1, (0, 255, 0), 10)
    if len(needToSeg) < 10:
        return [[gray_]]
    wholeAreaRect = cv2.boundingRect(maxAreaCNT)

    contWITHLOC = []
    for trangle in needToSeg:
        # [x, y]
        if trangle == wholeAreaRect:
            continue
        if trangle[0] < wholeAreaRect[0] or trangle[0] > wholeAreaRect[0] + wholeAreaRect[2] or \
                trangle[1] < wholeAreaRect[1] or trangle[1] > wholeAreaRect[1] + wholeAreaRect[3]:
            continue
        center = [trangle[0] + trangle[2] / 2, trangle[1] + trangle[3] / 2]
        i = 0
        havePlaced = False
        while not havePlaced:
            if i >= len(contWITHLOC):
                contWITHLOC.append([])
                contWITHLOC[i].append(trangle)
                havePlaced = True
                break
            if center[1] > contWITHLOC[i][0][1] and center[1] < contWITHLOC[i][0][1] + contWITHLOC[i][0][3] * 1:
                for j in range(len(contWITHLOC[i])):
                    xOFCNT = contWITHLOC[i][j][0] + contWITHLOC[i][j][2] / 2
                    if center[0] >= xOFCNT:
                        if j == len(contWITHLOC[i]) - 1:
                            contWITHLOC[i].insert(j + 1, trangle)
                            havePlaced = True
                            break
                        else:
                            continue
                    else:
                        contWITHLOC[i].insert(j, trangle)
                        havePlaced = True
                        break
            else:
                i += 1


    contWITHLOC = sorted(contWITHLOC, key=lambda contours : contours[0][1] + contours[0][3])
    result = []
    for i in range(len(contWITHLOC)):
        result.append([])
        for j in range(len(contWITHLOC[i])):
            x, y, w, h = contWITHLOC[i][j]
            roi = gray_[y : y + h, x: x + w]
            result[i].append(roi)
            cv2.imwrite(file_path_box_dir + str(i) + '__' + str(j) + '.jpg', roi)
    cv2.imwrite(file_path_box_dir + 'canny.jpg', canny)
    cv2.imwrite(file_path_box_dir + 'gray.jpg', gray)

    return result

def segmentChars(imgs, file_path_box_dir_char):
    if not os.path.exists(file_path_box_dir_char):
        os.mkdir(file_path_box_dir_char)
    # contours里面是存放的list,每个list存放处于同一行的图片块
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            img = imgs[i][j]
            img_new = img.copy()
            img_copy = img.copy()
            thisDir = file_path_box_dir_char + str(i) + '__' + str(j) + '/'
            if not os.path.exists(thisDir):
                os.mkdir(thisDir)
            textFile = open(thisDir + 'text.txt', 'w+')
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, kernel2)
            # img_new = cv2.GaussianBlur(img_new, (5, 5) ,0)
            # img_new = cv2.medianBlur(img_new, 5)
            img_new = cv2.adaptiveThreshold(img_new, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
            _, cnts, hierarchy = cv2.findContours(img_new, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            indexOfColor = 0
            newCnts = []
            for indexOfCont in range(len(cnts)):
                contour = cnts[indexOfCont]
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if area < 40 or area > 0.5 * img.shape[0] * img.shape[1]:
                    continue
                if max(h, w) / min(h, w) > 5:
                    continue
                # 最后的bool是为了下一步合并时，是否已经被合并过的标志
                newCnts.append([contour, [x, y, w, h], True])
            need_merge = True
            while need_merge:
                need_merge = False
                for m in range(len(newCnts)):
                    struct1 = newCnts[m]
                    x1, y1, w1, h1 = struct1[1]
                    contour1 = struct1[0]
                    if not struct1[2]:
                        continue
                    for n in range(len(newCnts)):
                        if m == n:
                            continue
                        struct2 = newCnts[n]
                        if struct2[2]:
                            contour2 = struct2[0]
                            x2, y2, w2, h2 = struct2[1]
                            # # 按照contour空间位置
                            if x1 <= x2:
                                x_left = x1
                                x_right = x2
                                y_left = y1
                                y_right = y2
                                w_left = w1
                                w_right = w2
                                h_left = h1
                                h_right = h2
                            else:
                                x_left = x2
                                x_right = x1
                                y_left = y2
                                y_right = y1
                                w_left = w2
                                w_right = w1
                                h_left = h2
                                h_right = h1
                            # # 右侧的contour的左上顶点在左侧的contour内部 对应 1、4、6、7
                            if (x_right <= x_left + w_left) and (y_right >= y_left and y_right <= y_left + h_left):
                                x1 = x_left
                                y1 = y_left
                                w1 = max(x_right + w_right - x_left, w_left)
                                h1 = max(y_right + h_right - y_left, h_left)
                                struct1[1] = [x1, y1, w1, h1]
                                struct2[2] = False
                                need_merge = True
                                break
                            # 右侧的contour的左下顶点在左侧的contour内部 对应 2、5
                            if (x_right <= x_left + w_left) and (y_right + h_right >= y_left and y_right + h_right <= y_left + h_left) and (y_right <= y_left):
                                x1 = x_left
                                y1 = y_right
                                w1 = max(x_right + w_right - x_left, w_left)
                                h1 = max(y_left + h_left - y_right, h_right)
                                struct1[1] = [x1, y1, w1, h1]
                                struct2[2] = False
                                need_merge = True
                                break
                            # # 左侧的contour的右上、右下顶点都在右面的contour内部 对应3、8
                            if (x_right <= x_left + w_left) and (y_right <= y_left) and (y_right + h_right >= y_left + h_left):
                                x1 = x_left
                                y1 = y_right
                                w1 = max(x_right + w_right - x_left, w_left)
                                h1 = h_right
                                struct1[1] = [x1, y1, w1, h1]
                                struct2[2] = False
                                need_merge = True
                                break
                            # 两个contour左右并列，并且纵轴方向有很大重叠部分
                            if (x_right - (x_left + w_left)) in range(8):
                                # 对应 1
                                if y_left >= y_right:
                                    dis = y_right + h_right - y_left
                                    if dis / h_left > 0.5 or dis / h_right > 0.5:
                                        x1 = x_left
                                        y1 = y_right
                                        w1 = x_right + w_right - x_left
                                        h1 = max(y_left + h_left - y_right, h_right)
                                        struct1[1] = [x1, y1, w1, h1]
                                        struct2[2] = False
                                        need_merge = True
                                        break
                                # # 对应 2
                                else:
                                    dis = y_left + h_left - y_right
                                    if dis / h_left > 0.5 or dis / h_right > 0.5:
                                        x1 = x_left
                                        y1 = y_left
                                        w1 = x_right + w_right - x_left
                                        h1 = max(y_right + h_right - y_left, h_left)
                                        struct1[1] = [x1, y1, w1, h1]
                                        struct2[2] = False
                                        need_merge = True
                                        break
                            # 对应 3
                            if ((y_left - (y_right + h_right)) in range(5)) and (x_right + w_right < x_left + w_left):
                                x1 = x_left
                                y1 = y_right
                                w1 = w_left
                                h1 = y_left + h_left - y_right
                                struct1[1] = [x1, y1, w1, h1]
                                struct2[2] = False
                                need_merge = True
                                break
                            # 对应 4
                            if ((y_right - (y_left + h_left)) in range(5)) and (x_right + w_right < x_left + w_left):
                                x1 = x_left
                                y1 = y_left
                                w1 = w_left
                                h1 = y_right + h_right - y_left
                                struct1[1] = [x1, y1, w1, h1]
                                struct2[2] = False
                                need_merge = True
                                break
            # # 此时应该调用波峰波谷算法，再次对所有的contour进行分割
            result_Rows = []
            method = 'find_gray_avg'
            # 需要重新检测contour
            for struct in newCnts:
                if struct[2]:
                    x_loc, y_loc, width, height = struct[1]
                    # 过小，作为噪声处理
                    if width < 10 or height < 10:
                        continue
                    if width* height < 300:
                        continue
                    # # 需要分割的情况
                    this_block = img_copy[y_loc:y_loc + height, x_loc:x_loc + width]

                    # 是否需要分割的FLAG
                    have_rows = False
                    if height > 60 and width > 50:
                        y = 0
                        rows = []
                        rows_gray_sum = 0
                        step_row = 1

                        # 计算每一行的平均像素值
                        while y + step_row < height:
                            row_num = {}
                            row_num["y"] = y
                            region = this_block[y:y + step_row, 0:width]
                            target_dev = cv2.meanStdDev(region)
                            target_gray_avg = target_dev[1][0][0]
                            row_num["num"] = target_gray_avg
                            rows_gray_sum += target_gray_avg
                            rows.append(row_num)
                            y += step_row

                        rows2 = [] # 存放各个行

                        if method == 'find_gray_avg':
                            # 使用平均灰度计算

                            # 找到像素均值低于平均的那些行，结合为几个部分
                            rows_gray_avg = rows_gray_sum * 1.005 / len(rows)
                            for row in rows:
                                row_num = row["num"]
                                row_y = row["y"]
                                if row_num > rows_gray_avg:
                                    row2_len = len(rows2)
                                    if row2_len == 0:
                                        row2 = {}
                                        row2["begin"] = row_y
                                        row2["end"] = row_y + step_row
                                        rows2.append(row2)
                                    else:
                                        # 根据当前的行号与上一行对比，看是否能否连接起来
                                        last_row = rows2[row2_len - 1]
                                        last_row_end = last_row["end"]
                                        if row_y == last_row_end:
                                            last_row["end"] = row_y + step_row
                                        else:
                                            # 不属于同一行
                                            row2 = {}
                                            row2["begin"] = row_y
                                            row2["end"] = row_y + step_row
                                            rows2.append(row2)
                        # 将比较小的row合并到一起，此时并没有考虑行之间的连续性
                        addition = 0
                        for indexOfRows in range(len(rows2)):
                            row = rows2[indexOfRows]
                            have_rows = True
                            row_begin = row["begin"]
                            row_end = row["end"]
                            if row_end - row_begin < 30:
                                if addition == 0:
                                    addition = row_end - row_begin
                                    continue
                                else:
                                    row_begin -= addition
                                    addition = 0
                            else:
                                addition = 0
                            result_Rows.append([x_loc, y_loc + row_begin - 3, width, row_end - row_begin + 3])
                            img = cv2.rectangle(img, (x_loc, y_loc + row_begin - 3), (x_loc + width, y_loc + row_end + 3), (0, 255, 0), 2)


                    # img = cv2.drawContours(img, struct[0], -1, colors[indexOfColor % 3], 1)
                    # indexOfColor += 1
                    if not have_rows:
                        result_Rows.append([x_loc, y_loc, width, height])
                        img = cv2.rectangle(img, (x_loc, y_loc), (x_loc + width, y_loc + height), (0, 255, 0), 2)

            # # tesseract识别，识别率及其低下
            # for x_loc, y_loc, region_width, region_height in result_Rows:
            #     region_col = img_copy[y_loc:y_loc + region_height, x_loc:x_loc + region_width]
            #     region_col = cv2PIL(region_col)
            #     text = pytesseract.image_to_string(region_col, lang='chi_sim+eng')
            #     textFile.write(str(x_loc) + ' ' + str(y_loc) + ' : ' + text + '\n')

            print('Doing ' + str(i) + ' ' + str(j))
            for x_loc, y_loc, region_width, region_height in result_Rows:
                region_col = img_new[y_loc:y_loc + region_height, x_loc:x_loc + region_width]
                _, region_col = cv2.threshold(region_col, 100, 255, cv2.THRESH_BINARY_INV)
                vertical_sum = np.sum(region_col, axis=0)
                if i == 2 and j == 2:
                    print(" ")
                if vertical_sum is None:
                    continue
                vertical_peek_ranges = extract_peek_ranges_from_array(
                    vertical_sum,
                    minimun_val=40,
                    minimun_range=1)
                # 如果没有进行分割
                if len(vertical_peek_ranges) == 0:
                    vertical_peek_ranges.append((0, region_width))
                elif len(vertical_peek_ranges) > 1:
                    # 处理一些过于小的部分
                    new_vertical_peek_ranges = []
                    need_jump = False
                    for loc, now_box in enumerate(vertical_peek_ranges):
                        if need_jump:
                            need_jump = False
                            continue
                        w = now_box[1] - now_box[0]
                        if w < 10:
                            if loc == 0:
                                next_box = vertical_peek_ranges[loc + 1]
                                new_vertical_peek_ranges.append([now_box[0], next_box[1]])
                                need_jump = True
                            elif loc == len(vertical_peek_ranges) - 1:
                                new_vertical_peek_ranges[-1][1] = now_box[1]
                            else:
                                pre_box = vertical_peek_ranges[loc - 1]
                                next_box = vertical_peek_ranges[loc + 1]
                                if pre_box[1] - pre_box[0] > next_box[1] - next_box[0]:
                                    new_vertical_peek_ranges.append([now_box[0], next_box[1]])
                                    need_jump = True
                                else:
                                    new_vertical_peek_ranges[-1][1] = now_box[1]
                        else:
                            new_vertical_peek_ranges.append(now_box)

                    vertical_peek_ranges = new_vertical_peek_ranges

                # 如果有剩余部分
                if region_width - vertical_peek_ranges[-1][1] > 50:
                    vertical_peek_ranges.append((vertical_peek_ranges[-1][1], region_width))

                for vertical_range in vertical_peek_ranges:
                    x = x_loc + vertical_range[0]
                    y = y_loc
                    w = vertical_range[1] - vertical_range[0]
                    h = region_height
                    pt1 = (x + 2, y + 2)
                    pt2 = (x + w - 2, y + h - 2)
                    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)




            cv2.imwrite(thisDir + 'contours.jpg', img)
            cv2.imwrite(thisDir + 'threshold.jpg', img_new)
            textFile.close()


    pass
