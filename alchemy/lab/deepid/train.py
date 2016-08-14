import os
from alchemy.models.symbol import get_net_symbol
from alchemy.train import train_model
from alchemy.train.options import args
import mxnet as mx

net = get_net_symbol(args.network, args.num_classes, args.use_bn)

# data
def get_iterator(args, kv):
    shapes = list(map(int, args.data_shape.strip().split(',')))
    data_shape = (3, shapes[0], shapes[1])
    train = mx.io.ImageRecordIter(
        path_imgrec        = os.path.join(args.data_dir, args.train_dataset),
        data_shape         = data_shape,
        batch_size         = args.batch_size,
        rand_mirror        = True,
        shuffle            = True,
        preprocess_threads = 4,
        num_parts          = kv.num_workers,
        part_index         = kv.rank)

    val = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.val_dataset),
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
