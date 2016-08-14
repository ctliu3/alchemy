import mxnet as mx

def get_symbol(num_classes, use_bn=True):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(4, 4), stride=(1, 1), num_filter=20)
    if use_bn:
        conv1 = mx.symbol.BatchNorm(data=conv1, name='bn1')
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(2, 2), stride=(2,2))

    # stage 2
    conv2 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), stride=(1, 1), num_filter=40)
    if use_bn:
        conv2 = mx.symbol.BatchNorm(data=conv2, name='bn2')
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(2, 2), stride=(1, 1), pool_type="max")

    # stage 3
    conv3 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=60)
    if use_bn:
        conv3 = mx.symbol.BatchNorm(data=conv3, name='bn3')
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu3, kernel=(2, 2), stride=(2, 2), pool_type="max")

    conv4 = mx.symbol.Convolution(
        data=pool3, kernel=(2, 2), pad=(1, 1), num_filter=80)
    if use_bn:
        conv4 = mx.symbol.BatchNorm(data=conv4, name='bn4')
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")

    flatten_pool3 = mx.symbol.Flatten(data=pool3)
    flatten_relu4 = mx.symbol.Flatten(data=relu4)

    # stage 4
    fc160_1 = mx.symbol.FullyConnected(data=flatten_pool3, num_hidden=160)
    fc160_2 = mx.symbol.FullyConnected(data=flatten_relu4, num_hidden=160)
    # multi-scale feature layer
    fc160 = mx.symbol.Concat(*[fc160_1, fc160_2])

    dropout = mx.symbol.Dropout(data=fc160, p=0.5)
    fc = mx.symbol.FullyConnected(data=dropout, num_hidden=num_classes)
    # stage 6
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax
