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

    for trial in range(args.trial):
        print("=============================Trial: {}=============================".format(trial + 1))
        train_dst, unlabeled_dst, test_dst = get_dataset(args, trial)

        # Initialize a labeled dataset by randomly sampling K=1,000 points from the entire dataset.
        I_index, O_index, U_index, Q_index = [], [], [], []
        I_index, O_index, U_index = get_sub_train_dataset(train_dst, args.target_list, I_index, O_index, U_index,
                                                          Q_index, args.n_query, args.ood_rate, initial=True)
        test_I_index = get_sub_test_dataset(test_dst, args.target_list)

        # TODO: Use DataLoaderX for ImageNet
        sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
        train_loader = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size, num_workers=args.workers)

        sampler_test = SubsetSequentialSampler(test_I_index)
        test_loader = DataLoader(test_dst, sampler=sampler_test, batch_size=args.test_batch_size, num_workers=args.workers)

        dataloaders = {'train': train_loader, 'test': test_loader}

        # Active learning cycles
        logs = []
        for cycle in range(args.cycle):
            print("====================Cycle: {}====================".format(cycle + 1))
            # Model
            print("| Training on model %s" % args.model)
            models = get_models(args, nets, args.model)
            torch.backends.cudnn.benchmark = False

            # Loss, criterion and scheduler (re)initialization
            criterion, optimizers, schedulers = get_optim_configurations(args, models)

            # Self-supervised Training (for CCAL and MQ-Net with CSI)
            models = self_sup_train(args, models, optimizers, schedulers, train_dst, I_index, O_index, U_index)

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

            I_index, O_index, U_index, in_cnt = get_sub_train_dataset(train_dst, args.target_list, I_index, O_index, U_index, Q_index,
                                                                       args.n_query, args.ood_rate, initial=False)

            print("# Labeled_in: {}, # Labeled_ood: {}, # Unlabeled: {}".format(
                len(set(I_index)), len(set(O_index)), len(set(U_index))))

            # TODO: training MQNET
            if args.method == 'MQNet':
                delta_loader = DataLoader(train_dst, sampler=SubsetRandomSampler(Q_index), batch_size=max(1, args.csi_batch_size), num_workers=args.workers)
                models = meta_train(args, models, optimizers, schedulers, criterion, dataloaders['train'], delta_loader)

            # Create a new dataloader for the updated labeled dataset
            sampler_labeled = SubsetRandomSampler(I_index)  # make indices initial to the samples
            dataloaders['train'] = DataLoader(train_dst, sampler=sampler_labeled, batch_size=args.batch_size)

            # save logs
            logs.append([acc, in_cnt])

        print("====================Logs, Trial {}====================".format(trial + 1))
        logs = np.array(logs).reshape((-1, 2))
        print(logs, flush=True)

'''
    # Training settings
    args = parser.parse_args()
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:' + str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    print("args: ", args)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_u_all, dst_test = datasets.__dict__[
        args.dataset](args)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
    print("im_size: ", dst_train[0][0].shape)

    # Initialize Unlabeled Set & Labeled Set
    indices = list(range(len(dst_train)))
    random.shuffle(indices)

    labeled_set = indices[:args.n_query]
    unlabeled_set = indices[args.n_query:]

    dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
    print("Initial set size: ", len(dst_subset))

    # BackgroundGenerator for ImageNet to speed up dataloaders
    if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
        train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                   pin_memory=False)
        test_loader = DataLoaderX(dst_test, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=False)
    else:
        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=False)

    # Get Model
    print("| Training on model %s" % args.model)
    network = get_model(args, nets, args.model)

    # Active learning cycles
    logs = []
    for cycle in range(args.cycle):
        print("====================Cycle: {}====================".format(cycle + 1))

        # Get optim configurations for Distrubted SGD
        criterion, optimizer, scheduler, rec = get_optim_configurations(args, network, train_loader)

        # Training
        print("==========Start Training==========")
        for epoch in range(args.epochs):
            # train for one epoch
            train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=False)

        acc = test(test_loader, network, criterion, epoch, args, rec)
        print('Cycle {}/{} || Label set size {}: Test acc {}'.format(cycle + 1, args.cycle, len(labeled_set), acc))

        # write logs
        logs.append([acc])
        if cycle == args.cycle - 1:
            break

        # AL Query Sampling
        print("==========Start Querying==========")

        selection_args = dict(selection_method=args.uncertainty,
                              balance=args.balance,
                              greedy=args.submodular_greedy,
                              function=args.submodular,
                              )
        ALmethod = methods.__dict__[args.method](dst_u_all, unlabeled_set, network, args, **selection_args)
        Q_indices, Q_scores = ALmethod.select()

        # Update the labeled dataset and the unlabeled dataset, respectively
        for idx in Q_indices:
            labeled_set.append(idx)
            unlabeled_set.remove(idx)

        print("# of Labeled: {}, # of Unlabeled: {}".format(len(labeled_set), len(unlabeled_set)))
        assert len(labeled_set) == len(list(set(labeled_set))) and len(unlabeled_set) == len(list(set(unlabeled_set)))

        # Re-Configure Training of the Next Cycle
        network = get_model(args, nets, args.model)

        dst_subset = torch.utils.data.Subset(dst_train, labeled_set)
        if args.dataset == "ImageNet" or args.dataset == "ImageNet30":
            train_loader = DataLoaderX(dst_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                       pin_memory=False)
        else:
            train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.workers, pin_memory=False)
    print("Final acc logs")
    logs = np.array(logs).reshape((-1, 1))
    print(logs, flush=True)
'''