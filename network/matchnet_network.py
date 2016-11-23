import mxnet as mx
import symbol_resnet as sr
def get_vgg_feature_conv(data, conv_weight, conv_bias):
    #cdata = data
    cdata = (data - 128)/160.
    cdata = mx.symbol.Convolution(data=cdata, kernel=(7,7), num_filter=96,pad=(3,3),
                                   weight = conv_weight[0], bias = conv_bias[0],
                                   name = 'conv0')
    cdata = mx.symbol.Activation(data=cdata,act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Convolution(data=cdata, kernel=(5, 5), num_filter=256,pad=(2,2),
                                  weight=conv_weight[1], bias=conv_bias[1],
                                  name='conv1')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=512,pad=(1,1),
                                  weight=conv_weight[2], bias=conv_bias[2],
                                  name='conv2')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=512,pad=(1,1),
                                  weight=conv_weight[3], bias=conv_bias[3],
                                  name='conv3')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=512,pad=(1,1),
                                  weight=conv_weight[4], bias=conv_bias[4],
                                  name='conv4')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Flatten(data=cdata)
    return cdata
def get_feature_conv(data, conv_weight, conv_bias):
    #cdata = data
    cdata = (data - 128)/160.
    cdata = mx.symbol.Convolution(data=cdata, kernel=(7,7), num_filter=24,pad=(3,3),
                                   weight = conv_weight[0], bias = conv_bias[0],
                                   name = 'conv0')
    cdata = mx.symbol.Activation(data=cdata,act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Convolution(data=cdata, kernel=(5, 5), num_filter=64,pad=(2,2),
                                  weight=conv_weight[1], bias=conv_bias[1],
                                  name='conv1')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=96,pad=(1,1),
                                  weight=conv_weight[2], bias=conv_bias[2],
                                  name='conv2')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=96,pad=(1,1),
                                  weight=conv_weight[3], bias=conv_bias[3],
                                  name='conv3')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Convolution(data=cdata, kernel=(3, 3), num_filter=64,pad=(1,1),
                                  weight=conv_weight[4], bias=conv_bias[4],
                                  name='conv4')
    cdata = mx.symbol.Activation(data=cdata, act_type="relu")
    cdata = mx.symbol.Pooling(data=cdata, pool_type="max", kernel=(3, 3), stride=(2, 2))
    cdata = mx.symbol.Flatten(data=cdata)
    return cdata
def get_match_net_(connect_type,feature_type):
    left_data = mx.symbol.Variable('left_data')
    right_data = mx.symbol.Variable('right_data')
    label = mx.symbol.Variable('label')
    feature_weight = []
    feature_bias = []
    for i in range(5):
        feature_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        feature_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    if feature_type == 'vgg_s':
        left_f = get_vgg_feature_conv(left_data, feature_weight, feature_bias)
        right_f = get_vgg_feature_conv(right_data, feature_weight, feature_bias)
    elif feature_type == 'match':
        left_f = get_feature_conv(left_data, feature_weight, feature_bias)
        right_f = get_feature_conv(right_data, feature_weight, feature_bias)
    left_flanten = mx.symbol.Flatten(data=left_f)
    right_flanten = mx.symbol.Flatten(data=right_f)
    if connect_type == 'concat':
        concate_feature = mx.symbol.Concat(left_flanten, right_flanten)
    elif connect_type == 'element_multi':
        concate_feature = left_flanten* right_flanten
    elif connect_type == 'element_plus':
        concate_feature = left_flanten + right_flanten
    fc1 = mx.symbol.FullyConnected(data=concate_feature, num_hidden=512)
    fc1_relu = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1_relu, num_hidden=512)
    fc2_relu = mx.symbol.Activation(data=fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax', label=label)
    return softmax
def get_match_net():
    left_data = mx.symbol.Variable('left_data')
    right_data = mx.symbol.Variable('right_data')
    label = mx.symbol.Variable('label')
    feature_weight = []
    feature_bias = []
    for i in range(5):
        feature_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        feature_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    left_f = get_feature_conv(left_data,feature_weight,feature_bias)
    right_f = get_feature_conv(right_data,feature_weight,feature_bias)
    left_flanten = mx.symbol.Flatten(data=left_f)
    right_flanten = mx.symbol.Flatten(data=right_f)
    concate_feature = mx.symbol.Concat(left_flanten,right_flanten)
    fc1 = mx.symbol.FullyConnected(data=concate_feature,num_hidden=512)
    fc1_relu = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1_relu, num_hidden=512)
    fc2_relu = mx.symbol.Activation(data=fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2)
    #softmax = mx.symbol.softmax_cross_entropy(fc3,label,name='softmax')
    softmax = mx.symbol.SoftmaxOutput(data=fc3,name='softmax',label=label)
    return softmax
def get_match_plus_net():
    left_data = mx.symbol.Variable('left_data')
    right_data = mx.symbol.Variable('right_data')
    label = mx.symbol.Variable('label')
    feature_weight = []
    feature_bias = []
    for i in range(5):
        feature_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        feature_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    left_f = get_feature_conv(left_data,feature_weight,feature_bias)
    right_f = get_feature_conv(right_data,feature_weight,feature_bias)
    left_flanten = mx.symbol.Flatten(data=left_f)
    right_flanten = mx.symbol.Flatten(data=right_f)
    #concate_feature = mx.symbol.Concat(left_flanten,right_flanten)
    plus_feature = left_flanten+right_flanten
    fc1 = mx.symbol.FullyConnected(data=plus_feature,num_hidden=512)
    fc1_relu = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1_relu, num_hidden=512)
    fc2_relu = mx.symbol.Activation(data=fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2)
    #softmax = mx.symbol.softmax_cross_entropy(fc3,label,name='softmax')
    softmax = mx.symbol.SoftmaxOutput(data=fc3,name='softmax',label=label)
    return softmax
def get_match_innerproduct_net():
    left_data = mx.symbol.Variable('left_data')
    right_data = mx.symbol.Variable('right_data')
    label = mx.symbol.Variable('label')
    feature_weight = []
    feature_bias = []
    for i in range(5):
        feature_weight.append(mx.sym.Variable('conv' + str(i) + '_weight'))
        feature_bias.append(mx.sym.Variable('conv' + str(i) + '_bias'))
    left_f = get_feature_conv(left_data,feature_weight,feature_bias)
    right_f = get_feature_conv(right_data,feature_weight,feature_bias)
    left_flanten = mx.symbol.Flatten(data=left_f)
    right_flanten = mx.symbol.Flatten(data=right_f)
    #concate_feature = mx.symbol.Concat(left_flanten,right_flanten)
    plus_feature = left_flanten*right_flanten
    fc1 = mx.symbol.FullyConnected(data=plus_feature,num_hidden=512)
    fc1_relu = mx.symbol.Activation(data=fc1, act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=fc1_relu, num_hidden=512)
    fc2_relu = mx.symbol.Activation(data=fc2, act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=fc2_relu, num_hidden=2)
    #softmax = mx.symbol.softmax_cross_entropy(fc3,label,name='softmax')
    softmax = mx.symbol.SoftmaxOutput(data=fc3,name='softmax',label=label)
    return softmax