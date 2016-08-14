from os.path import expanduser
import os
import sys
HOME = expanduser("~")
sys.path.insert(0, os.path.join(HOME, 'allblue/codes/mxnet/python'))

NET_MODULE = 'alchemy.lab.%s'

NETWORK_SYMBOL = {
    'alexnet': NET_MODULE % 'alexnet.symbol_alexnet',
    'deepid': NET_MODULE % 'deepid.symbol_deepid',
    # 'squeeze-net': 'lab.squeeze-net.symbol_squeezenet',
    # 'squeeze-net-bn': 'lab.squeeze-net.symbol_squeezenet_bn',
    # 'inception-v1': 'lab.inception-v1.symbol_inception-v1',
    # 'inception-bn': 'alchemy.lab.bn.symbol_inception-bn',
}

import importlib
def get_net_symbol(net_name, num_classes, use_bn=False):
    symbol_net = NETWORK_SYMBOL[net_name]
    net = importlib.import_module(symbol_net).get_symbol(num_classes, use_bn)
    return net
