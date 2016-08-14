import os
from random import shuffle
from alchemy.dataflow.yfd import YFD

import argparse

parser = argparse.ArgumentParser(description='split train/val set on face dataset')
parser.add_argument('--dataset', type=str, default='YFD',
                    choices=['LFW', 'YFD', 'webface', 'celeba'],
                    help = 'the dataset to use')
parser.add_argument('--data-dir', type=str,
                    help='image data directory')
parser.add_argument('--shuffle', default=True, action='store_true',
                    help='shuffle the face image in each class')
parser.add_argument('--train-ratio', type=float,
                    help='train ratio on each class')
parser.add_argument('--num-train-val', type=int,
                    help='number of face images used for training and validation')
parser.add_argument('--num-trains', type=int,
                    help='number of face images used for training')
parser.add_argument('--output-dir', type=str,
                    help='directory that store the train/val data')
args = parser.parse_args()

DATA_FETCHER = {
    'YFD': YFD,
}

def main():
    print(args.data_dir)
    assert os.path.isdir(args.data_dir)
    assert os.path.isdir(args.output_dir)

    DB = DATA_FETCHER[args.dataset]
    train_val_set = DB.aligned_train_val(args.data_dir, shuffle=True,
                                         num_train_val=args.num_train_val,
                                         num_trains=args.num_trains,
                                         train_ratio=args.train_ratio)

    train_set = []
    val_set = []
    for cid, data_set in train_val_set.items():
        for fn in data_set['train']:
            train_set.append((cid, fn))
        for fn in data_set['val']:
            val_set.append((cid, fn))

    cnt = 0
    shuffle(train_set)
    fout_train = open(os.path.join(args.output_dir, 'train.lst'), 'w')
    for cid, fn in train_set:
        fout_train.write('{}\t{}\t{}\n'.format(cnt, cid, fn))
        cnt += 1

    shuffle(val_set)
    fout_val = open(os.path.join(args.output_dir, 'val.lst'), 'w')
    for cid, fn in val_set:
        fout_val.write('{}\t{}\t{}\n'.format(cnt, cid, fn))
        cnt += 1

if __name__ == '__main__':
    main()
