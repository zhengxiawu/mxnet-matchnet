import cv2
import numpy as np
from facepp import File
from facepp import API
import shutil
from im2rec import *
def mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
API_KEY = '1aa3014184b76bcc27d3e5e2b399b82d'
API_SECRET = 'kIRfkyzOw3xByCNeV5qQunbHaVHEQUIN'
api = API(API_KEY,API_SECRET)
source_image_dir = '/home/sherwood/data/face/'
dst_face_dir = '/home/sherwood/data/face_cut_opencv/'
image_file_list = ['.png','.jpg','.JPG','.JPEG','.bmp']
detect_image_path_list = ['Facescrub_seg']
face_cascade = cv2.CascadeClassifier("/home/sherwood/tools/opencv/data/haarcascades/haarcascade_frontalface_default.xml")
count = 0
for i in detect_image_path_list:
    dst_face_sub_dir = os.path.join(dst_face_dir,i)
    mkdir(dst_face_sub_dir)
    src_face_sub_dir = os.path.join(source_image_dir,i)
    for dirs in os.listdir(src_face_sub_dir):
        mkdir(os.path.join(dst_face_sub_dir,dirs))
        for fname in os.listdir(os.path.join(src_face_sub_dir,dirs)):
            suffix = os.path.splitext(fname)[1].lower()
            if suffix in image_file_list:
                file_name_ = os.path.join(src_face_sub_dir,dirs,fname)
                dst_file_name = os.path.join(dst_face_sub_dir,dirs,fname)
                if not os.path.isfile(dst_file_name):
                    img = cv2.imread(file_name_)
                    if not img is None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                        if len(faces)==1:
                            for (x, y, w, h) in faces:
                                opencv_face_image = img[y:y+h,x:x+w]
                                cv2.imwrite(dst_file_name,opencv_face_image)
                                count+=1
                                print count
                    # else:
                    #     try:
                    #         result = api.detection.detect(img= File(file_name_))
                    #         if len(result['face']) == 1:
                    #             face = result['face'][0]['position']
                    #             x = int(face['center']['x'] * result['img_width'] * 0.01)
                    #             y = int(face['center']['y'] * result['img_height'] * 0.01)
                    #             h = int(face['height'] * result['img_height'] * 0.01 * 1.2)
                    #             w = int(face['width'] * result['img_width'] * 0.01 * 1.2)
                    #
                    #             img = img[y - h / 2:y + h / 2, x - w / 2:x + w / 2]
                    #             # cv2.rectangle(img,(x-w/2,y-h/2),(x+w/2,y+h/2),(0,255,0),2)
                    #             cv2.imwrite(os.path.join(dst_face_sub_dir, dirs, fname), img)
                    #             print count
                    #             count += 1
                    #     except:
                    #         break
# face_cascade = cv2.CascadeClassifier("/home/sherwood/tools/opencv/data/haarcascades/haarcascade_lefteye_2splits.xml")
#
# image = cv2.imread("/home/sherwood/data/face/imdb_crop/00/nm0000100_rm197368064_1955-1-6_2003.jpg")
#
# dst = cv2.resize(image,(300,300),interpolation=cv2.INTER_CUBIC)
# gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray,1.2,5)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
#
# cv2.imshow('test',dst)
# cv2.waitKey(0)
