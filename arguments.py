import argparse
import numpy as np
import time
from utils import *

parser = argparse.ArgumentParser(description='Parameter Processing')

# Basic arguments
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset') #CIFAR10, CIFAR100, ImageNet50
parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--gpu', default=None, nargs="+", type=int, help='GPU id to use')
parser.add_argument("--data-parallel", default=False, type=str_to_bool, help="whether parallel or not")
parser.add_argument('--ood-rate', type=float, default=0.6, metavar='N', help='OOD rate in unlabeled set')
parser.add_argument('--n-class', type=str, default=10, help='# of classes')
parser.add_argument('--trial', type=int, default=5, help='# of runs')
parser.add_argument('--cycle', type=int, default=10, help='# of AL cycles')
parser.add_argument('--n-query', type=int, default=1000, help='# of query samples')
parser.add_argument('--subset', type=int, default=50000, help='subset')
parser.add_argument('--resolution', type=int, default=32, help='resolution') # 32
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--print_freq', '-p', default=300, type=int, help='print frequency (default: 20)')
parser.add_argument('--seed', default=0, type=int, help="random seed")
parser.add_argument('-j', '--workers', default=5, type=int, help='number of data loading workers (default: 4)')
parser.add_argument("--ssl-save", default=True, type=str_to_bool, help="whether save ssl model or not")

# Optimizer and scheduler
parser.add_argument('--optimizer', default="SGD", help='optimizer to use, e.g. SGD, Adam')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for updating network parameters')
parser.add_argument('--lr-mqnet', type=float, default=0.001, help='learning rate for updating mqnet')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)', dest='weight_decay')
parser.add_argument("--scheduler", default="MultiStepLR", type=str, help="Learning rate scheduler") #CosineAnnealingLR, StepLR, MultiStepLR
parser.add_argument('--min_lr', type=float, default=1e-4, help='minimum learning rate for CosineAnnealingLR')
parser.add_argument("--gamma", type=float, default=.5, help="Gamma value for StepLR")
parser.add_argument("--step_size", type=float, default=50, help="Step size for StepLR")
parser.add_argument('--milestone', type=list, default=[100, 150], metavar='M', help='Milestone for MultiStepLR')
parser.add_argument('--warmup', type=int, default=10, metavar='warmup', help='warmup epochs')

# Training
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--epoch-loss', default=120, type=int, help='number of epochs for training loss module in LL')
parser.add_argument('--epochs-ccal', default=700, type=int, help='number of epochs for training contrastive coders in CCAL')
parser.add_argument('--epochs-csi', default=1000, type=int, help='number of epochs for training CSI')
parser.add_argument('--epochs-mqnet', default=100, type=int, help='number of epochs for training mqnet')
parser.add_argument('--steps-per-epoch', type=int, default=100, metavar='N', help='number of steps per epoch')
parser.add_argument('--batch-size', "-b", default=64, type=int, metavar='N')
parser.add_argument("--test-batch-size", "-tb", default=500, type=int)
parser.add_argument('--ccal-batch-size', default=32, type=int, metavar='N')
parser.add_argument('--csi-batch-size', default=32, type=int, metavar='N')

# Testing
parser.add_argument("--test_interval", '-ti', default=1, type=int, help=
"the number of training epochs to be preformed between two test epochs; a value of 0 means no test will be run (default: 1)")
parser.add_argument("--test_fraction", '-tf', type=float, default=1.,
                    help="proportion of test dataset used for evaluating the model (default: 1.)")

# AL Algorithm
parser.add_argument('--method', default="Uncertainty", help="specifiy AL method to use") #Uncertainty, Coreset, LL, BADGE, CCAL, SIMILAR, MQNet
parser.add_argument('--submodular', default="logdetcmi", help="specifiy submodular function to use") #flcmi, logdetcmi
parser.add_argument('--submodular_greedy', default="LazyGreedy", help="specifiy greedy algorithm for submodular optimization")
parser.add_argument('--uncertainty', default="CONF", help="specifiy uncertanty score to use") #CONF, Margin, Entropy
# for CCAL
parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',default=0.08, type=float)
parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',action='store_true')
parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',default=1.0, type=float)
parser.add_argument('--shift_trans_type', help='shifting transformation type', default='rotation',choices=['rotation', 'cutperm', 'none'], type=str)
parser.add_argument("--ood_samples", help='number of samples to compute OOD score',default=1, type=int)
parser.add_argument('--k', help='Initial learning rate', default=100.0, type=float)
parser.add_argument('--t', help='Initial learning rate', default=0.9, type=float)
# for MQNet
parser.add_argument('--mqnet-mode', default="CONF", help="specifiy the mode of MQNet to use") #CONF, LL

# Checkpoint and resumption
parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')
parser.add_argument('--resume', '-r', type=str, default='', help="path to latest checkpoint (default: do not load)")

args = parser.parse_args()