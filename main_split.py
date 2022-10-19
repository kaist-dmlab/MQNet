# Python
import os
import time
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Utils
from utils import *

# Custom
from arguments import parser
from load_split_data import get_dataset, get_sub_train_dataset, get_sub_test_dataset

import nets

import methods as methods

# Seed
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Main
if __name__ == '__main__':
    # Training settings
    args = parser.parse_args()
    args = get_more_args(args)
    print("args: ", args)

    # Runs on Different Class-splits
    for trial in range(args.trial):
        print("=============================Trial: {}=============================".format(trial + 1))
        train_dst, unlabeled_dst, test_dst = get_dataset(args, trial)

        # Initialize a labeled dataset by randomly sampling K=1,000 points from the entire dataset.
        I_index, O_index, U_index, Q_index = [], [], [], []
        I_index, O_index, U_index = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=True)
        test_I_index = get_sub_test_dataset(args, test_dst)

        # DataLoaders
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            sampler_test = SubsetSequentialSampler(test_I_index)
            train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)
        elif args.dataset == 'ImageNet50': # DataLoaderX for efficiency
            dst_subset = torch.utils.data.Subset(train_dst, I_index)
            dst_test = torch.utils.data.Subset(test_dst, test_I_index)
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
            test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Active learning
        logs = []
        for cycle in range(args.cycle):
            print("====================Cycle: {}====================".format(cycle + 1))
            # Model (re)initialization
            print("| Training on model %s" % args.model)
            models = get_models(args, nets, args.model)
            torch.backends.cudnn.benchmark = False

            # Loss, criterion and scheduler (re)initialization
            criterion, optimizers, schedulers = get_optim_configurations(args, models)

            # Self-supervised Training (for CCAL and MQ-Net with CSI)
            if cycle == 0:
                models = self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index)

            # Training
            t = time.time()
            train(args, models, criterion, optimizers, schedulers, dataloaders)
            print("cycle: {}, elapsed time: {}".format(cycle, (time.time() - t)))

            # Test
            acc = test(args, models, dataloaders)

            print('Trial {}/{} || Cycle {}/{} || Labeled IN size {}: Test acc {}'.format(
                    trial + 1, args.trial, cycle + 1, args.cycle, len(I_index), acc), flush=True)

            #### AL Query ####
            print("==========Start Querying==========")
            selection_args = dict(I_index=I_index,
                                  O_index=O_index,
                                  selection_method=args.uncertainty,
                                  dataloaders=dataloaders,
                                  cur_cycle=cycle)

            ALmethod = methods.__dict__[args.method](args, models, unlabeled_dst, U_index, **selection_args)
            Q_index, Q_scores = ALmethod.select()

            I_index, O_index, U_index, in_cnt = get_sub_train_dataset(args, train_dst, I_index, O_index, U_index, Q_index, initial=False)

            print("# Labeled_in: {}, # Labeled_ood: {}, # Unlabeled: {}".format(
                len(set(I_index)), len(set(O_index)), len(set(U_index))))

            # Only for MQNet
            if args.method == 'MQNet':
                delta_loader = DataLoader(train_dst, sampler=SubsetRandomSampler(Q_index), batch_size=max(1, args.batch_size), num_workers=args.workers)
                models = meta_train(args, models, optimizers, schedulers, criterion, dataloaders['train'], delta_loader)

            # Update trainloader
            if args.dataset in ['CIFAR10', 'CIFAR100']:
                sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
                dataloaders['train'] = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)
            elif args.dataset == 'ImageNet50':
                dst_subset = torch.utils.data.Subset(train_dst, I_index)
                train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)

            # save logs
            logs.append([acc, in_cnt])

        print("====================Logs, Trial {}====================".format(trial + 1))
        logs = np.array(logs).reshape((-1, 2))
        print(logs, flush=True)

        file_name = 'logs/'+str(args.dataset)+'/r'+str(args.ood_rate)+'_t'+str(trial)+'_'+str(args.method)
        if args.method == 'MQNet':
            file_name = file_name+'_'+str(args.mqnet_mode)
        np.savetxt(file_name, logs, fmt='%.4f', delimiter=',')