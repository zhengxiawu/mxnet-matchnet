import shutil
import os
import cv2
def mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
def return_img_num(name_list,image_file_list):
    count = 0
    for i in name_list:
        suffix = os.path.splitext(i)[1]
        if suffix in image_file_list:
            count+=1
    return count
class_name = 0
image_num = 0
root_dir = '/home/sherwood/data/face_cut_opencv/'
dst_dir = '/home/sherwood/data/face_cut_opencv_64/'
dirs_ = ['CACD2000_seg','CASIA_WebFace','Facescrub_seg','MSRA-CFW']
image_file_list = ['.png','.jpg','.JPG','.JPEG','.bmp']
for i in dirs_:
    parenets_dir_path = os.path.join(root_dir,i)
    dst_parents_dir_path = os.path.join(dst_dir,i)
    mkdir(dst_parents_dir_path)
    a = os.listdir(parenets_dir_path)
    for j in a:
        dst_dir_path = os.path.join(dst_parents_dir_path,j)
        mkdir(dst_dir_path)
        dir_path = os.path.join(parenets_dir_path,j)
        b = os.listdir(dir_path)
        image_num_ = return_img_num(b,image_file_list)
        image_num+=image_num_
        if image_num_>2:
            for k in b:
                img = cv2.imread(os.path.join(dir_path,k))
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img,)
                pass