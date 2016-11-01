import cv2
face_cascade = cv2.CascadeClassifier("/home/sherwood/tools/opencv/data/haarcascades/haarcascade_lefteye_2splits.xml")

image = cv2.imread("/home/sherwood/data/face/imdb_crop/00/nm0000100_rm197368064_1955-1-6_2003.jpg")

dst = cv2.resize(image,(300,300),interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.2,5)
print len(faces)
for (x,y,w,h) in faces:
    cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('test',dst)
cv2.waitKey(0)