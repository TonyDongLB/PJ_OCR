#!/usr/bin/env python3
#-*-encoding:utf-8-*-

import os,sys
from PIL import Image
import subprocess
import cv2
import numpy as np
#处理横竖问题
def change_wh(src_path,out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    out_dir_path = os.path.dirname(out_path)
    if not os.path.exists(out_dir_path) :
        os.makedirs(out_dir_path)
    #logger.debug("out_path:"+out_path)
    img = cv2.imread(src_path)

    img_height = img.shape[0]
    img_width = img.shape[1]
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_new = cv2.adaptiveThreshold(img_new, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
    img_new, cnts, hierarchy = cv2.findContours(img_new, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_new = cv2.cvtColor(img_new, cv2.COLOR_GRAY2BGR)
    area_map = {}
    index=0

    for c in cnts:
        area = cv2.contourArea(c)
        area_map[area] = c
        #img_new = cv2.rectangle(img_new,(x,y),(x+w,y+h),(255,0,0))
    sort_keys = sorted(area_map.keys())
    c = area_map.get(sort_keys[-2])
    (x, y, w, h) = cv2.boundingRect(c)
    #print("wh======(x, y, w, h):",(x, y, w, h))
    img_new = cv2.rectangle(img_new, (x, y), (x + w, y + h), (255, 0, 0),10)
    cv2.imwrite(out_path+"_01.jpg", img_new)


    #logger.debug("width:",img_width)
    #logger.debug("height:",img_height)
#如果长宽比错误,进行90度的处理
    #if img_height>img_width :
    if h>w :
        #M = cv2.getRotationMatrix2D((img_height/2,img_width/2), 90, 1)
        pts1 = np.float32([[0, 0], [img_width, 0], [0, img_height]])
        pts2 = np.float32([[img_width, 0], [img_width, img_height], [0, 0]])
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (img_width,img_height))
        img = cv2.resize(img, (img_height,img_width ), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(out_path,img)

def change_tb(img,out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    out_dir_path = os.path.dirname(out_path)
    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
    # 利用章的信息来确定票据的正反
    B, G, R = cv2.split(img)
    R = R - B - G

    retval, R_thre = cv2.threshold(R, thresh= 200, maxval= 255, type= cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    R_thre = cv2.morphologyEx(R_thre, cv2.MORPH_OPEN, kernel)
    R_thre = cv2.GaussianBlur(R_thre, (7, 7), 0)

    #找到面积最大的两个最大轮廓
    _, cnts, hierarchy = cv2.findContours(R_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxAreaCnt = cnts[0]
    nextMaxAreaCnt = cnts[1]
    R_thre = cv2.cvtColor(R_thre, cv2.COLOR_GRAY2BGR)
    for contour in cnts:
        if len(contour) < 5:
            continue
        thisArea = cv2.contourArea(contour)
        if thisArea > cv2.contourArea(maxAreaCnt):
            maxAreaCnt = contour
        else:
            if thisArea > cv2.contourArea(nextMaxAreaCnt):
                nextMaxAreaCnt = contour

    isEllipse = []
    for contour in [maxAreaCnt, nextMaxAreaCnt]:
        ellipse = cv2.fitEllipse(contour)
        thisArea = cv2.contourArea(contour)
        ellipseArea = np.pi * np.abs(ellipse[0][0] - ellipse[0][1]) * np.abs(ellipse[1][0] - ellipse[1][1])
        differ = np.abs(ellipseArea - thisArea)
        if differ / thisArea < 0.2:
            isEllipse.append(True)
        else:
            isEllipse.append(False)
    #先验知识：票据原本的stamp比后来的小
    if isEllipse[1] == True:
        stamp = nextMaxAreaCnt
    else:
        stamp = maxAreaCnt

    x, y, w, h = cv2.boundingRect(stamp)
    R_thre = cv2.rectangle(R_thre,(x, y), (x + w, y + h), (0, 255, 0), 20)


    R_thre = cv2.drawContours(R_thre, maxAreaCnt, -1, (0, 255, 0), 5)
    R_thre = cv2.drawContours(R_thre, nextMaxAreaCnt, -1, (255, 0, 0), 5)
    center = (x + w / 2, y + h / 2)
    height = img.shape[0]
    width = img.shape[1]
    if center[0] > 0.45 * width and center[0] < 0.55 * width:
        if center[1] > 0.6 * height:
            if height > width:
                img = cv2.copyMakeBorder(img, 0, 0, height - width, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                maxLength = height
            if width > height:
                img = cv2.copyMakeBorder(img, width - height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                maxLength = width
            PerspectiveMatrix = cv2.getRotationMatrix2D((maxLength / 2, maxLength / 2), 180, 1)
            img = cv2.warpAffine(img, PerspectiveMatrix, (maxLength, maxLength))
            img = img[: height, :width]
        else:
            pass
    else:
        if center[1] > 0.45 * height and center[1] < 0.55 * height:
            if center[0] < 0.4 * width:
                if height > width:
                    img = cv2.copyMakeBorder(img, 0, 0, 0, height - width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    maxLength = height
                if width > height:
                    img = cv2.copyMakeBorder(img, width - height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    maxLength = width
                PerspectiveMatrix = cv2.getRotationMatrix2D((maxLength / 2, maxLength / 2), 270, 1)
                img = cv2.warpAffine(img, PerspectiveMatrix, (maxLength, maxLength))
                img = img[: width, :height]
            if center[0] > 0.6 * width:
                if height > width:
                    img = cv2.copyMakeBorder(img, 0, 0, height - width, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    maxLength = height
                if width > height:
                    img = cv2.copyMakeBorder(img, 0, width - height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    maxLength = width
                PerspectiveMatrix = cv2.getRotationMatrix2D((maxLength / 2, maxLength / 2), 90, 1)
                img = cv2.warpAffine(img, PerspectiveMatrix, (maxLength, maxLength))
                img = img[: width, :height]
        else:
            pass


    cv2.imwrite(out_path + '____R___.jpg', R_thre)
    cv2.imwrite(out_path, img)
    return img

    #可以检测到contour，之后是最小的外接椭圆，如果面积相差不多，则认为其是椭圆~~~
    pass

