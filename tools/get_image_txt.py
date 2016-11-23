import os,random
root_dir = '/home/sherwood/data/face_cut_opencv_64_/CASIA_WebFace/'
train_txt_file_path = '/home/sherwood/data/face_cut_opencv_64_/casia_train.txt'
test_txt_file_path = '/home/sherwood/data/face_cut_opencv_64_/casia_test.txt'
train_txt = open(train_txt_file_path,'w')
test_txt = open(test_txt_file_path,"w")
image_file_list = ['.png','.jpg','.JPG','.JPEG','.bmp']
a = os.listdir(root_dir)
count = 0
class_name = 0
for i in a:
    image_dir = os.path.join(root_dir,i)
    b = os.listdir(image_dir)
    if random.random() < 0.9:
        for j in b:
            suffix = os.path.splitext(j)[1]
            if suffix in image_file_list:
                image_file_path = os.path.join(image_dir,j)
                train_txt.write(image_file_path + " " + str(class_name) + '\n')
                count+=1
    else:
        for j in b:
            suffix = os.path.splitext(j)[1]
            if suffix in image_file_list:
                image_file_path = os.path.join(image_dir, j)
                test_txt.write(image_file_path + " " + str(class_name) + '\n')
                count += 1
    class_name+=1
    print count