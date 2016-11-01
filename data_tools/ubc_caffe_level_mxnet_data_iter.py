import random,sys,leveldb
import mxnet as mx
import math
sys.path.append('/home/sherwood/project/caffe/python/')
import caffe
from caffe.proto import caffe_pb2
def load_from_caffe_leveldb(data_name):
    db = leveldb.LevelDB('/home/sherwood/project/matchnet/data/leveldb/'+data_name+'.leveldb/')
    #db = leveldb.LevelDB('/home/sherwood/project/matchnet/data/leveldb/test.leveldb/')
    #db = leveldb.LevelDB('/home/sherwood/project/mxnet-matchnet/data/test.leveldb/')
    img_data = []
    label_data = []
    count = 0
    for k in db.RangeIter():
        datum = caffe_pb2.Datum.FromString(k[1])
        array = caffe.io.datum_to_array(datum)
        label = datum.label
        img_data.append(array)
        label_data.append(label)
        count +=1
    return img_data,label_data
class Batch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class DateIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.batch_size = data_shapes[0][0]
        self.cur_batch = 0
        self.unique_label,self.unique_label_index = self.get_unique_set_and_index()
        self.data_length = len(self.unique_label)
        self.num_batches = math.ceil(self.data_length / self.batch_size)
    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def get_unique_set_and_index(self):
        unique_label = list(set(self.label_gen))
        unique_label_index = [[]]*len(unique_label)
        for i in range(len(self.label_gen)):
            if len(unique_label_index[self.label_gen[i]]) == 0:
                unique_label_index[self.label_gen[i]] = [i]
            else:
                unique_label_index[self.label_gen[i]].append(i)
        # unique_label_index = []
        # label_flag = unique_label[0]
        # start_ind = 0
        # for i in range(len(unique_label)):
        #     temp = []
        #     while label_flag == self.label_gen[start_ind]:
        #         temp.append(start_ind)
        #         start_ind += 1
        #         if start_ind >= self.data_length:
        #             break
        #     if len(temp) < 2:
        #         StopIteration
        #     unique_label_index.append(temp)
        #     label_flag = self.label_gen[start_ind] if start_ind < self.data_length else 0
        # # length check
        # if not len(unique_label) == len(unique_label_index):
        #     StopIteration
        return unique_label,unique_label_index
    #generate the data base
    def generate_data(self):
        #generate random pairs in paper MatchNet
        positive_num = self.batch_size/2
        negative_num = self.batch_size/2
        label = []
        batch_left_img = []
        batch_right_img = []
        count = 0
        start_ind = self.cur_batch*self.batch_size
        end_ind = min((self.cur_batch+1)*self.batch_size,len(self.unique_label))
        #generate positive set
        for i in range(start_ind,end_ind):
            if count<positive_num:
                #add positive sample in batch
                label.append(1)
                random_sample = random.sample(self.unique_label_index[i],2)
                batch_left_img.append(self.data_gen[random_sample[0]])
                batch_right_img.append(self.data_gen[random_sample[1]])
                count+=1
            else:
                random_sample = random.sample(range(len(self.label_gen)),2)
                neg_count = 0
                while self.label_gen[random_sample[0]] == self.label_gen[random_sample[1]] and neg_count<1000:
                    neg_count+=1
                    random_sample = random.sample(range(len(self.label_gen)), 2)
                label.append(0)
                batch_left_img.append(self.data_gen[random_sample[0]])
                batch_right_img.append(self.data_gen[random_sample[1]])
        while not end_ind-start_ind == 128:
            random_sample = random.sample(range(len(self.label_gen)), 2)
            neg_count = 0
            while self.label_gen[random_sample[0]] == self.label_gen[random_sample[1]] and neg_count < 1000:
                neg_count += 1
                random_sample = random.sample(range(len(self.label_gen)), 2)
            label.append(0)
            batch_left_img.append(self.data_gen[random_sample[0]])
            batch_right_img.append(self.data_gen[random_sample[1]])
            end_ind+=1
        return label,batch_left_img,batch_right_img


    def next(self):
        if self.cur_batch < self.num_batches:
            label, batch_left_img, batch_right_img = self.generate_data()
            self.cur_batch += 1
            return Batch([mx.nd.array(batch_left_img),mx.nd.array(batch_right_img)], [mx.nd.array(label)])
        else:
            raise StopIteration