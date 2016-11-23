import shutil
import os
import cv2
import random
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
dst_dir = '/home/sherwood/data/face_cut_opencv_64_/'
mkdir(dst_dir)
train_txt_image_file = '/home/sherwood/data/face_cut_opencv_64_/train.txt'
test_txt_image_file = '/home/sherwood/data/face_cut_opencv_64_/test.txt'
train_txt = open(train_txt_image_file,'w')
test_txt = open(test_txt_image_file,"w")
dirs_ = ['CACD2000_seg','CASIA_WebFace','Facescrub_seg','MSRA-CFW']
image_file_list = ['.png','.jpg','.JPG','.JPEG','.bmp']
for i in dirs_:
    parenets_dir_path = os.path.join(root_dir,i)
    dst_parents_dir_path = os.path.join(dst_dir,i)
    mkdir(dst_parents_dir_path)
    a = os.listdir(parenets_dir_path)
    for j in a:
        dst_dir_path = os.path.join(dst_parents_dir_path,j)
        dir_path = os.path.join(parenets_dir_path,j)
        b = os.listdir(dir_path)
        image_num_ = return_img_num(b,image_file_list)
        if image_num_>2:
            mkdir(dst_dir_path)
            if random.random()<0.9:
                for k in b:
                    img = cv2.imread(os.path.join(dir_path,k))
                    if img is not None:
                        # if len(img.shape) == 3:
                        #     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img,(64,64))
                        cv2.imwrite(os.path.join(dst_dir_path,k),img)
                        train_txt.write(os.path.join(dst_dir_path,k)+" "+str(class_name)+'\n')
                        image_num+=1
                        print image_num
            else:
                for k in b:
                    img = cv2.imread(os.path.join(dir_path, k))
                    if img is not None:
                        # if len(img.shape) == 3:
                        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (64, 64))
                        cv2.imwrite(os.path.join(dst_dir_path, k), img)
                        test_txt.write(os.path.join(dst_dir_path, k) + " " + str(class_name)+ '\n')
                        image_num += 1
                        print image_num
            class_name+=1
