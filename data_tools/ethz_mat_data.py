import h5py
import mxnet as mx
import random
import math
root_path = '/home/sherwood/project/mxnet-matchnet/'
def load_ethz_data(data_name,train_test):
    file = h5py.File(root_path+'data/ethz_mat/'+data_name+'/'+train_test+'.mat')
    if train_test == 'train':
        train_data = file['head_train_data'][:]
        train_label = file['head_train_labels'][:]
    else:
        train_data = file['head_test_data'][:]
        train_label = file['head_test_labels'][:]
    train_label = train_label[0,:]
    return_data = []
    return_label = []
    for i in range(train_label.size):
        return_data.append(train_data[i,:,:,:])
        return_label.append(int(train_label[i]))
    return return_data,return_label
class Batch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class DateIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=100):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.batch_size = data_shapes[0][0]
        self.cur_batch = 0
        self.unique_label,self.unique_label_index = self.get_unique_set_and_index()
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
        unique_label_index = []
        for label in unique_label:
            unique_label_index.append([i for i,x in enumerate(self.label_gen) if x == label])
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
        for i in range(self.batch_size):
            if count < positive_num:
                count+=1
                # add positive sample in batch
                label.append(1)
                random_sample_index = random.sample(self.unique_label_index,1)
                random_sample = random.sample(random_sample_index[0],2)
                batch_left_img.append(self.data_gen[random_sample[0]])
                batch_right_img.append(self.data_gen[random_sample[1]])
            else:
                random_sample = random.sample(range(len(self.label_gen)), 2)
                neg_count = 0
                while self.label_gen[random_sample[0]] == self.label_gen[random_sample[1]] and neg_count < 1000:
                    neg_count += 1
                    random_sample = random.sample(range(len(self.label_gen)), 2)
                label.append(0)
                batch_left_img.append(self.data_gen[random_sample[0]])
                batch_right_img.append(self.data_gen[random_sample[1]])
                count += 1
        return label,batch_left_img,batch_right_img


    def next(self):
        if self.cur_batch < self.num_batches:
            label, batch_left_img, batch_right_img = self.generate_data()
            self.cur_batch += 1
            return Batch([mx.nd.array(batch_left_img),mx.nd.array(batch_right_img)], [mx.nd.array(label)])
        else:
            raise StopIteration