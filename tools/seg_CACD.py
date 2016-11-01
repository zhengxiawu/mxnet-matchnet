import shutil
import os
import cv2
def mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
root = '/home/sherwood/data/face/CACD2000/'
dst = '/home/sherwood/data/face/CACD2000_seg/'
image_file_list = ['.png','.jpg','.JPG','.JPEG','.bmp']
count = 0
for fname in os.listdir(root):
    fpath = os.path.join(root,fname)
    preffix = os.path.splitext(fname)[0]
    preffix = preffix[0:-5]
    mkdir(os.path.join(dst,preffix))
    suffix = os.path.splitext(fname)[1].lower()
    if suffix in image_file_list:
        shutil.copy(fpath,os.path.join(dst,preffix,fname))
        print count
        count+=1
