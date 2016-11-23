import mxnet as mx
from network import matchnet_network as matn
from data_tools import face_data_set as fds
if __name__=='__main__':
    load_model = True 
    save_model_prefix = 'paramters/face_matchnet_casia_vgg_s_color/face_matchnet_'
    #read data from caffe
    batch_size = 128
    #load the data and initialize the dataIter
    test_data = fds.DateIter(['left_data', 'right_data'],
                             [(batch_size, 3, 64, 64), (batch_size, 3, 64, 64)], 'casia_test',
                             ['label'], [(batch_size,)])
    data = fds.DateIter(['left_data', 'right_data'],
                  [(batch_size,3,64,64), (batch_size,3,64,64)],'casia_train',
                  ['label'], [(batch_size,)])

    #set GPU
    devs = [mx.gpu(1)]
    #initialize optimizer
    sgd_opt = mx.optimizer.SGD(learning_rate=0.01,momentum=0.0,wd=0.005,rescale_grad=(1.0/batch_size))
    #set the model prefix
    checkpoint = mx.callback.do_checkpoint(save_model_prefix,period=10)
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
        epoch = 90
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
        network = matn.get_match_net_('concat','vgg_s')
        model = mx.model.FeedForward(
            ctx=devs,
            symbol=network,
            num_epoch=1000,
            optimizer= sgd_opt,
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            )
    model.fit(
        X=data,
        eval_data=test_data,
        eval_metric=eval_metrics,
        epoch_end_callback=epoch_call,
        batch_end_callback=batch_call
    )
