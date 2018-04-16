#planA 通过长宽比例切出对应内容的图片
import cv2
import sys,os
import numpy as np
import threading
import img_process.utils.resize as resize
import img_process.utils.change as change
import img_process.utils.check as check
import img_process.utils.box as box
import datetime
import threadpool
import time
import re
import PIL.Image as Image

import json
import  math
import shutil

width = 2048

def img_process(filepath, config):
    begin = datetime.datetime.now()
    data_dir = config["data_dir"]
    file_dir_img_root = data_dir
    img = {}
    re_str = '.*/(.*)\.(.*)'
    re_pat = re.compile(re_str)
    search_ret = re_pat.search(filepath)
    if search_ret:
        file_name = search_ret.groups()[0]
        file_suffix = search_ret.groups()[1]
        img = {}
        img['name'] = file_name
        img['suffix'] = file_suffix
        img['path'] = filepath
    file_name = img["name"]
    file_suffix = img["suffix"]

    # 原图
    file_path_src = file_dir_img_root + file_name + '.' + file_suffix
    #储存一系列的图片的路径
    file_path_src_copy = file_dir_img_root + file_name + '/000_src_copy_' + file_name + '.' + file_suffix
    # 横竖调整图
    file_path_change_wh = file_dir_img_root + file_name + '/001_change_wh_' + file_name + '.' + file_suffix
    # 检测发票表格图
    file_path_rect = file_dir_img_root + file_name + '/002_rect_' + file_name + '.' + file_suffix
    # 大小调整
    file_path_resize = file_dir_img_root + file_name + '/003_resize_' + file_name + '.' + file_suffix
    # 灰度二值图
    # file_path_gray = file_dir_img_root + file_name + '/gray' + file_name + '.' + file_suffix
    # 上下调整图
    file_path_change_tb = file_dir_img_root + file_name + '/004_change_tb_' + file_name + '.' + file_suffix

    file_path_box_dir = file_dir_img_root + file_name + '/box'

    if not os.path.exists(file_path_src):
        print('图片:' + file_path_src +' 不存在，请检查！')
        return

    img = cv2.imread(file_path_src)
    if img.size < 480000:
        print("图片分辨率太低,至少大于800*600!")
        return
    else:
        if not os.path.exists(file_dir_img_root + file_name):
            os.makedirs(file_dir_img_root + file_name)

        cv2.imwrite(file_path_src_copy, img)
        if img.shape[1] * img.shape[0] > 2000 * 3000:
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            cv2.imwrite(file_path_src, img)

    check_result = check.DL_check_rect(file_path_src, file_path_rect)
    # 大小重置
    resize.resize(file_path_rect, file_path_resize, width)
    change.change_tb(file_path_resize, file_path_change_tb)






    pass