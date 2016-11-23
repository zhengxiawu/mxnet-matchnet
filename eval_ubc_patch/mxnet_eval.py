import mxnet as mx
import sys
import numpy as np
from eval_metrics import *
from data_tools import ubc_test_data_iter as ubc
sys.path.append('/home/sherwood/project/caffe/python/')
def main():
    import os
    test_dir_name = 'yosemite'
    prefix = '/home/sherwood/project/mxnet-matchnet/paramters/match_net/match_net'
    num_round = 103
    model = mx.model.FeedForward.load(prefix,num_round,ctx=mx.gpu(1),numpy_batch_size = 1024)
    # prob = model.predict(test_data)[0]
    test_data_iter = ubc.DateIter(test_dir_name,1000)
    prob = model.predict(test_data_iter)
    scores = prob[:,1]
    np.save('yosemite_result_1.npy',scores)
    error_at_95 = ErrorRateAt95Recall(test_data_iter.labels, scores)
    print "Error rate at 95%% recall: %0.2f%%" % (error_at_95 * 100)
    # Compute matching prediction.
if __name__ == '__main__':
    main()