import sys,os,io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

import base64


def get_base64(file):
    data = None
    with open(file, 'rb') as f:
        data = base64.b64encode(f.read())  # 读取文件内容，转换为base64编码
    return dataz

import json
#public params

#获取当前运行脚本的绝对路径
py_dir = os.path.dirname(os.path.realpath(__file__))
print("py_dir = " + py_dir)
project_dir = os.path.dirname(py_dir)
print(u"加入到PATH的是 project_dir = "  + project_dir)
sys.path.append(project_dir)

import api_invoice.service.DL_img as img

config = {"data_dir" : project_dir + '/imgs/'}
upload_path = config['data_dir']
allImgNames = []
for filename in os.listdir(upload_path):
    if filename == '.DS_Store':
        continue
    else:
        if os.path.isdir(os.path.join(upload_path, filename)):
            continue
        allImgNames.append(filename)
result = {}
for filename in allImgNames:
    filepath =  os.path.join(upload_path, filename)
    print('现在正在处理图片：' + filepath)
    result[filename] = img.img_process(filepath, config)




