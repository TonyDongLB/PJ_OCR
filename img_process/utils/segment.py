import os,sys
import cv2
import numpy as np
import math

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
        pass
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
            thisDir = file_path_box_dir_char + str(i) + '__' + str(j) + '/'
            if not os.path.exists(thisDir):
                os.mkdir(thisDir)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            img_new = cv2.morphologyEx(img_new, cv2.MORPH_OPEN, kernel2)
            img_new = cv2.medianBlur(img_new, 7)
            img_new = cv2.adaptiveThreshold(img_new, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 5)
            _, cnts, hierarchy = cv2.findContours(img_new, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            indexOfColor = 0
            newCnts = []
            for indexOfCont in range(len(cnts)):
                contour = cnts[indexOfCont]
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if area < 50 or area > 0.2 * img.shape[0] * img.shape[1]:
                    continue
                if max(h, w) / min(h, w) > 4:
                    continue
                #最后的bool是为了下一步合并时，是否已经被合并过的标志
                newCnts.append( (contour, (x, y, w, h), True))
            for m in range(len(newCnts)):
                struct1 = newCnts[m]
                x1, y1, w1, h1 = struct1[1]
                contour1 = struct1[0]
                for n in range(len(newCnts)):
                    struct2 = newCnts[n]
                    if struct2[2]:
                        contour2 = struct2[0]
                        x2, y2, w2, h2 = struct2[1]
                        if x2 > x1 and y2 > y1 :
                            # contour2在contour1内部
                            if x2 + w2 < x1 + w1 and y2 + h2 < y1 + h1:
                                struct2[2] = False
                                break
                            # contour2的一部分在contour1内部
                            else:
                                cv2.line(img_new, contour1[0][0], contour2[0][0], (255), 2)
                                struct2[2] = False
                                break
                        # if
            for struct in newCnts:
                if struct[2]:
                    img = cv2.drawContours(img, struct[0], -1, colors[indexOfColor % 3], 1)
                    indexOfColor += 1
                    x, y, w, h = struct[1]
                    img = cv2.rectangle(img ,(x, y), (x + w, y + h), (0, 255, 0), 2)







            cv2.imwrite(thisDir + 'contours.jpg', img)
            cv2.imwrite(thisDir + 'threshold.jpg', img_new)


    pass
