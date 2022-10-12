import os
import numpy as np
import torch
import math
import random
from torch.utils.data.dataset import Subset
from datasets.cifar10 import MyCIFAR10
from datasets.cifar100 import MyCIFAR100
from torchvision import datasets
import torchvision.transforms as T

CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],#1  d
    [1, 33, 67, 73, 91],#2  d
    [54, 62, 70, 82, 92],#3
    [9, 10, 16, 29, 61],#4
    [0, 51, 53, 57, 83],#5
    [22, 25, 40, 86, 87],#6
    [5, 20, 26, 84, 94],#7
    [6, 7, 14, 18, 24],#8   d
    [3, 42, 43, 88, 97],#9   d
    [12, 17, 38, 68, 76],#10
    [23, 34, 49, 60, 71],#11
    [15, 19, 21, 32, 39],#12  d
    [35, 63, 64, 66, 75],#13  d
    [27, 45, 77, 79, 99],#14
    [2, 11, 36, 46, 98],#15
    [28, 30, 44, 78, 93],#16   d
    [37, 50, 65, 74, 80],#17   d
    [47, 52, 56, 59, 96],#18
    [8, 13, 48, 58, 90],#19
    [41, 69, 81, 85, 89],#20
]


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)

def get_subset_with_len(dataset, length, shuffle=False):
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset

def get_dataset(args, trial):
    # Transform
    T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100

    train_transform = T.Compose(
        [T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])  #
    test_transform = T.Compose([T.ToTensor(), T_normalize])

    if args.dataset == 'CIFAR10':
        # cifar10
        file_path = '../data/cifar10/'
        train_set = MyCIFAR10(file_path, train=True, download=False, transform=train_transform)
        unlabeled_set = MyCIFAR10(file_path, train=True, download=False, transform=test_transform)
        test_set = MyCIFAR10(file_path, train=False, download=False, transform=test_transform)
    elif args.dataset == 'CIFAR100':
        # cifar100
        file_path = '../data/cifar100/'
        train_set = MyCIFAR100(file_path, train=True, download=False, transform=train_transform)
        unlabeled_set = MyCIFAR100(file_path, train=True, download=False, transform=test_transform)
        test_set = MyCIFAR100(file_path, train=False, download=False, transform=test_transform)

    # for split
    if args.dataset == 'CIFAR10':
        args.num_images = 50000
        args.num_classes = 10
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 64
        args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3]]
        args.target_list = args.target_lists[trial]
        args.untarget_list = list(np.setdiff1d(list(range(0, 10)), list(args.target_list)))
        args.target_number = 4

    elif args.dataset == 'CIFAR100':
        args.num_images = 50000
        args.num_classes = 100
        args.input_size = 32 * 32 * 3
        args.batch_size_classifier = 64
        args.target_lists = [[33, 10, 74, 72, 88, 47, 27, 68, 60, 75, 45, 79, 92, 35, 86, 50, 18,
            61, 49, 29, 23, 30, 67, 73, 82, 94, 13, 37, 39, 26, 62, 22, 90, 53, 89, 11,  3, 20, 70, 96], \
                            [69,  8, 86, 18, 68, 30, 75,  3, 63, 76, 72,  7, 50, 81, 46, 89, 22,
            93, 62, 21, 33, 98, 82, 20, 60,  5, 77,  1, 74, 88, 57, 34, 43, 27, 66, 83, 25, 48,  4, 55], \
                            [70, 28, 60, 22, 39, 35, 73, 13, 74, 10,  2, 16, 80, 53, 67, 66, 78,
            46, 26, 71, 43, 38, 42, 14, 50, 77, 20, 48, 52,  8, 54, 58, 91,  5, 25, 90, 61, 11, 59, 55], \
                            [ 7, 93, 37, 84, 57, 99, 10, 75, 54, 42, 26, 27, 47, 52, 61, 86, 60,
            90,  1,  0, 98, 87, 94, 74, 56, 91, 23, 97, 30, 17, 53, 12, 76, 11, 25, 65, 96,  3, 45, 8], \
                            [ 0,  1,  4,  5,  7,  9, 12, 19, 21, 22, 23, 24, 38, 41, 42, 43, 46,
            47, 48, 51, 55, 59, 60, 62, 68, 73, 75, 78, 79, 80, 81, 85, 86, 90,91, 94, 95, 96, 97, 98]]# SEED 1
        args.target_list = args.target_lists[trial]
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
        args.target_number = 40 #20

    # class converting
    for i, c in enumerate(args.untarget_list):
        train_set.targets[np.where(train_set.targets == c)[0]] = int(args.num_classes)
        test_set.targets[np.where(test_set.targets == c)[0]] = int(args.num_classes)

    args.target_list.sort()
    for i, c in enumerate(args.target_list):
        train_set.targets[np.where(train_set.targets == c)[0]] = i
        test_set.targets[np.where(test_set.targets == c)[0]] = i

    train_set.targets[np.where(train_set.targets == int(args.num_classes))[0]] = int(args.target_number)
    test_set.targets[np.where(test_set.targets == int(args.num_classes))[0]] = int(args.target_number)

    unlabeled_set.targets = train_set.targets

    print("Target classes: ", args.target_list)

    uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
    print("Train, # samples per class")
    print(uni, cnt)
    uni, cnt = np.unique(np.array(test_set.targets), return_counts=True)
    print("Test, # samples per class")
    print(uni, cnt)

    return train_set, unlabeled_set, test_set

def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset

def get_sub_train_dataset(dataset, classes, L_index, O_index, U_index, Q_index, budget, ood_rate, initial= False):
    if initial:
        L_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
        O_total= [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] >= len(classes)]

        n_ood = round(len(L_total)*(ood_rate/(1-ood_rate)))
        O_total = random.sample(O_total, n_ood)

        print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))

        L_index = random.sample(L_total, int(budget*(1-ood_rate)))
        O_index = random.sample(O_total, int(budget*ood_rate))
        U_index = list(set(L_total + O_total)-set(L_index)-set(O_index))

        print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(len(L_index), len(O_index), len(U_index)))
        return L_index, O_index, U_index
    else:
        Q_index = list(Q_index)
        Q_label = [dataset[i][1] for i in Q_index]

        in_Q_index, ood_Q_index = [], []
        for i, c in enumerate(Q_label):
            if c < len(classes):
                in_Q_index.append(Q_index[i])
            else:
                ood_Q_index.append(Q_index[i])
        print("# query in: {}, ood: {}".format(len(in_Q_index), len(ood_Q_index)))

        L_index = L_index + in_Q_index
        O_index = O_index + ood_Q_index
        U_index = list(set(U_index) - set(Q_index))
        return L_index, O_index, U_index, len(in_Q_index)

def get_sub_test_dataset(dataset, classes):
    labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
    return labeled_index


def get_sub_unlabeled_dataset(dataset, select_L_index,select_O_index, target_list, num_images):
    all_index = set(np.arange(num_images))
    select_index = select_L_index + select_O_index
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)
    datasey_UL = Subset(dataset, unlabeled_L_index)
    datasey_UO = Subset(dataset, unlabeled_O_index)
    dataset_U = Subset(dataset, unlabeled_indices)

    return dataset_U, datasey_UL, datasey_UO, unlabeled_indices, unlabeled_L_index, unlabeled_O_index

def get_unlabeled_dataset(dataset, select_L_index, target_list,mismatch, num_images):

    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)

    target_number = len(unlabeled_L_index)
    others_number = math.ceil((mismatch*target_number)/(1-mismatch))

    select_O_index = random.sample(unlabeled_O_index, others_number)
    unlabeled_index = unlabeled_L_index + select_O_index
    #dataset_U = Subset(dataset, unlabeled_index)

    return unlabeled_index

def get_mismatch_contrast_dataset(dataset, select_L_index, target_list,mismatch, num_images):
    all_index = set(np.arange(num_images))
    unlabeled_indices = list(np.setdiff1d(list(all_index),select_L_index))  # find indices which is in all_indices but not in current_indices

    unlabeled_L_index = []
    unlabeled_O_index = []
    for i in unlabeled_indices:
        if dataset[i][1] in target_list:
            unlabeled_L_index.append(i)
        else:
            unlabeled_O_index.append(i)

    target_number = len(unlabeled_L_index)
    others_number = math.ceil((mismatch*target_number)/(1-mismatch))

    select_O_index = random.sample(unlabeled_O_index, others_number)
    unlabeled_index = unlabeled_L_index + select_O_index
    contrast_index = unlabeled_index + select_L_index

    random.shuffle(contrast_index)
    dataset_contrast = Subset(dataset, contrast_index)

    return dataset_contrast,contrast_index

def get_label_index(dataset, L_index,args):
    label_i_index = [[] for i in range(len(args.target_list))]
    for i in L_index:
        for k in range(len(args.target_list)):
            if dataset[i][1] == args.target_list[k]:
                label_i_index[k].append(i)
    return label_i_index