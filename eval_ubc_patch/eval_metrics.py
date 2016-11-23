"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.
"""
import operator
from data_tools import ubc_test_data_iter as ubc
import numpy as np

def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    test = sorted_scores.sort(key=operator.itemgetter(1), reverse=True)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    tp = 0
    count = 0
    for label, score in sorted_scores:
        print count
        count += 1
        if label == 1:
            tp += 1
        else:
            pass
        if tp >= n_thresh:
            break

    return float(count - tp) / count
if __name__ == '__main__':
    test_dir_name = 'notredame'
    test_data_iter = ubc.DateIter(test_dir_name, 1000)
    scores = np.load('result_1.npy')
    error_at_95 = ErrorRateAt95Recall(test_data_iter.labels, scores)