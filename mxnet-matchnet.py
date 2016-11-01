import mxnet as mx
from network import matchnet_network as matn
from data_tools import ubc_caffe_level_mxnet_data_iter as ubc
if __name__=='__main__':
    load_model = True
    #read data from caffe
    batch_size = 128
    #load the data and initialize the dataIter
    img_data, label_data = ubc.load_from_caffe_leveldb('liberty')
    data = ubc.DateIter(['left_data', 'right_data'],
                  [(batch_size,1,64,64), (batch_size,1,64,64)],
                    img_data,
                  ['label'], [(batch_size,)],
                    label_data)
    #set GPU
    devs = mx.gpu(0)
    #initialize optimizer
    sgd_opt = mx.optimizer.SGD(learning_rate=0.01,momentum=0.0,wd=0.0005,rescale_grad=(1.0/batch_size))
    #set the model prefix
    save_model_prefix = 'paramters/match_net/match_net'
    checkpoint = mx.callback.do_checkpoint(save_model_prefix,period=1)
    batch_call = []
    epoch_call = []
    batch_call.append(mx.callback.log_train_metric(1))
    epoch_call.append(checkpoint)
    def lr_callback(epoch, symbol, arg_params, aux_params):
        if epoch % 100 == 0 and not epoch == 0:
            sgd_opt.lr /= 10
            print 'now the learning rate is:' + str(sgd_opt.lr)
    epoch_call.append(lr_callback)
    eval_metrics = ['accuracy']
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    if load_model:
        epoch = 10
        network,arg_params,aux_params = mx.model.load_checkpoint(save_model_prefix,epoch)
        model = mx.model.FeedForward(
            ctx=devs,
            symbol=network,
            arg_params=arg_params,
            aux_params=aux_params,
            begin_epoch=epoch,
            num_epoch=1000,
            optimizer=sgd_opt,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        )
    else:
        network = matn.get_match_net()
        model = mx.model.FeedForward(
            ctx=devs,
            symbol=network,
            num_epoch=1000,
            optimizer= sgd_opt,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            )
    model.fit(
        X=data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_call,
        batch_end_callback=batch_call
    )
