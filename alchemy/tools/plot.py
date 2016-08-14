import sys
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import re
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='plot the curve')
parser.add_argument('--log-file', type=str,
                    help='log file to analyze')
parser.add_argument('--line-start', type=int, default=0,
                    help='start line to analyze')
parser.add_argument('--line-end', type=int, default=-1,
                    help='end line to analyze')
parser.add_argument('--plot-name', type=str, required=True,
                    help='filename to save the plot')
args = parser.parse_args()

EPOCH_PAT = re.compile('Epoch\[([\d]+)\]')
TRAIN_ACC_PAT = re.compile('.*?]\sTrain-accuracy=([\d\.]+)')
VAL_ACC_PAT = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')

class TrainAnalyst(object):

    epoch_map = defaultdict(dict)

    @classmethod
    def put(self, line):
        pass

class ValAnslyst(object):

    epoch_map = defaultdict(dict)

    @classmethod
    def put(self, line):
        acc = re.findall(VAL_ACC_PAT, line)
        if not acc:
            return
        epoch = re.findall(EPOCH_PAT, line)
        if not epoch:
            return
        epoch = int(epoch[0])
        acc = float(acc[0])

        self.epoch_map[epoch] = {'acc': acc}

    @classmethod
    def pprint(self):
        for epoch, value in self.epoch_map.iteritems():
            print('Epoch: {}, acc: {}'.format(epoch, value['acc']))

    @classmethod
    def accuracy(self):
        epochs, accuracys = [], []
        for epoch, value in self.epoch_map.iteritems():
            epochs.append(epoch)
            accuracys.append(value['acc'])

        return epochs, accuracys


if __name__ == '__main__':
    assert os.path.isfile(args.log_file)

    line_no = -1
    with open(args.log_file, 'r') as f:
        for line in f:
            line_no += 1
            if line_no < args.line_start or \
                    (args.line_end > 0 and args.line_end > line_no):
                continue
            line = line.strip()
            if line.find('Epoch') == -1:
                continue
            TrainAnalyst.put(line)
            ValAnslyst.put(line)

    epochs, accuracys = ValAnslyst.accuracy()
    plt.xlabel('epoch')
    plt.ylabel('val accuracy')
    plt.plot(epochs, accuracys)
    plt.savefig(args.plot_name)
