#-*- coding:utf-8 -*-
import sys
import leveldb
import numpy as np
sys.path.append('/home/sherwood/project/caffe/python/')
import caffe
from caffe.proto import caffe_pb2
import pickle
db = leveldb.LevelDB('/home/sherwood/project/matchnet/data/leveldb/liberty.leveldb/')
a = db.RangeIter()
img_data = []
label_data = []
count =1
for k in db.RangeIter():
    datum = caffe_pb2.Datum.FromString(k[1])
    array = caffe.io.datum_to_array(datum)
    label = datum.label
    img_data.append(array)
    label_data.append(label)
    print count
    count+=1
pickle.dump(img_data,open('/home/sherwood/project/mxnet-matchnet/data/liberty_img.txt',"w"))
pickle.dump(label_data,open('/home/sherwood/project/mxnet-matchnet/data/liberty_label.txt',"w"))
print "done"