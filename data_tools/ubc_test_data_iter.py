import mxnet as mx
import leveldb
import sys
import numpy as np
import math
from data_tools import ubc_caffe_level_mxnet_data_iter as ubc
sys.path.append('/home/sherwood/project/caffe/python/')
from caffe.proto import caffe_pb2
import os
import random
def ReadPairs(filename):
    """Read pairs and match labels from the given file.
    """
    pairs = []
    labels = []
    with open(filename) as f:
        for line in f:
            parts = [p.strip() for p in line.split()]
            pairs.append((parts[0], parts[3]))
            labels.append(1 if parts[1] == parts[4] else 0)

    return pairs, labels


def ReadPatches(db, pairs, patch_height=64, patch_width=64):
    """Read patches from the given db handle. Each element in pairs is a
    pair of keys.

    Returns
    -------
    Two N * 1 * W * H array in a list, where N is the number of pairs.
    """
    N = len(pairs)
    patches = [np.zeros((N, 1, patch_height, patch_width),
                        dtype=np.float),
               np.zeros((N, 1, patch_height, patch_width),
                        dtype=np.float)]
    idx = 0  # Index to the next available patch in the patch array.
    parity = 0
    for pair in pairs:
        for key in pair:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(db.Get(key))
            patches[parity][idx, 0, :, :] = \
                np.fromstring(datum.data, np.uint8).reshape(
                patch_height, patch_width)
            parity = 1 - parity

        idx += 1
    return patches
class Batch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class DateIter(mx.io.DataIter):
    def __init__(self, data_names, batch_size):
        self.root_dara_dir = '/home/sherwood/project/matchnet/data'
        self.test_pairs = os.path.join(self.root_dara_dir, 'phototour/' + data_names + '/' + 'm50_100000_100000_0.txt')
        self.test_db = self.root_dara_dir + '/leveldb/' + data_names + '.leveldb'
        self._provide_data = zip(['left_data', 'right_data'], [(batch_size,1,64,64), (batch_size,1,64,64)])
        self._provide_label = zip(['label'], [(batch_size,)])
        self.pairs, self.labels = ReadPairs(self.test_pairs)
        self.db =  leveldb.LevelDB(self.test_db, create_if_missing=False)
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.labels)/self.batch_size)
        print self.num_batches
        self.cur_batch = 0
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
    def generate_data(self):
        start_idx = self.cur_batch*self.batch_size
        stop_idx = (self.cur_batch+1)*self.batch_size
        input_patches = ReadPatches(self.db, self.pairs[start_idx:stop_idx])
        input_label = self.labels[start_idx:stop_idx]
        return input_label,input_patches[0],input_patches[1]
    def next(self):
        if self.cur_batch < self.num_batches:
            label, batch_left_img, batch_right_img = self.generate_data()
            self.cur_batch += 1
            print self.cur_batch
            return Batch([mx.nd.array(batch_left_img),mx.nd.array(batch_right_img)], [mx.nd.array(label)])
        else:
            raise StopIteration
