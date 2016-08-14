import argparse

parser = argparse.ArgumentParser(description='train an image classifer on dataset')

parser.add_argument('--network', type=str, required=True,
                    help = 'the neural network to use')
parser.add_argument('--data-dir', type=str, required=True,
                    help='the input data directory')
parser.add_argument('--save-model-prefix', type=str, required=True,
                    help='the prefix of the model to save')
parser.add_argument('--gpus', type=str, required=True,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, required=True,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, required=True,
                    help='the number of classes')
parser.add_argument('--batch-size', type=int, required=True,
                    help='the batch size')
parser.add_argument('--use-bn', type=str, choices=['True', 'False'], required=True,
                    help='whether to use batch norm')
parser.add_argument('--data-shape', type=str, required=True,
                    help='set shape of image')
parser.add_argument('--log-file', type=str, required=True,
                    help='the name of log file')

# Optional argument.
parser.add_argument('--model-prefix', type=str,
                    help='the prefix of the model to load')
parser.add_argument('--load-epoch', type=int,
                    help="load the model with specific epoch")
parser.add_argument('--lr', type=float, default=0.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.9,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--weight_decay', type=float, default=0.00001,
                    help='weight decay used for loss regularization')
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-dir', type=str, default="/tmp/",
                    help='directory of the log file')
parser.add_argument('--train-dataset', type=str, default="train.rec",
                    help='train dataset name')
parser.add_argument('--val-dataset', type=str, default="val.rec",
                    help="validation dataset name")
parser.add_argument('--metrics', type=str, default="top1",
                    help='metric after bench end')
parser.add_argument('--num-batch-callback', type=int, default=100,
                    help='number of batch training stop and callback')

args = parser.parse_args()
args.use_bn = True if args.use_bn == 'True' else False
