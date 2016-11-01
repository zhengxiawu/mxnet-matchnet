import mxnet as mx
import numpy as np
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class SimpleIter:
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
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

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            return SimpleBatch(data, label)
        else:
            raise StopIteration

num_users = 1000
num_items = 1000
k = 10
user = mx.symbol.Variable('user')
item = mx.symbol.Variable('item')
score = mx.symbol.Variable('score')
# user feature lookup
user = mx.symbol.Embedding(data = user, input_dim = num_users, output_dim = k)
# item feature lookup
item = mx.symbol.Embedding(data = item, input_dim = num_items, output_dim = k)
# predict by the inner product, which is elementwise product and then sum
pred = user * item
pred = mx.symbol.sum_axis(data = pred, axis = 1)
pred = mx.symbol.Flatten(data = pred)
# loss layer
pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
n = 32
data = SimpleIter(['user', 'item'],
                  [(n,), (n,)],
                  [lambda s: np.random.randint(0, num_users, s),
                   lambda s: np.random.randint(0, num_items, s)],
                  ['score'], [(n,)],
                  [lambda s: np.random.randint(0, 5, s)])

mod = mx.mod.Module(symbol=pred, data_names=['user', 'item'], label_names=['score'])
mod.fit(data, num_epoch=5)




