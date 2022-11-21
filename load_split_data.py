import pickle
import numpy as np
import torch
import math
import random
from torch.utils.data.dataset import Subset
from datasets.cifar10 import MyCIFAR10
from datasets.cifar100 import MyCIFAR100
from datasets.imagenet import MyImageNet
from torchvision import datasets
import torchvision.transforms as T

CIFAR10_SUPERCLASS = list(range(10))  # one class
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
IMAGENET_SUPERCLASS = list(range(30))  # one class

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
    # Normalization
    if args.dataset == 'CIFAR10':
        T_normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif args.dataset == 'CIFAR100':
        T_normalize = T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    elif args.dataset == 'ImageNet50':
        T_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Transform
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(size=32, padding=4), T.ToTensor(), T_normalize])  #
        test_transform = T.Compose([T.ToTensor(), T_normalize])
    elif args.dataset == 'ImageNet50':
        train_transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), T_normalize])
        test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T_normalize])

    # Dataset
    if args.dataset == 'CIFAR10':

        file_path = args.data_path + '/cifar10/'
        train_set = MyCIFAR10(file_path, train=True, download=True, transform=train_transform)
        unlabeled_set = MyCIFAR10(file_path, train=True, download=True, transform=test_transform)
        test_set = MyCIFAR10(file_path, train=False, download=True, transform=test_transform)
    elif args.dataset == 'CIFAR100':
        file_path = args.data_path + '/cifar100/'
        train_set = MyCIFAR100(file_path, train=True, download=True, transform=train_transform)
        unlabeled_set = MyCIFAR100(file_path, train=True, download=True, transform=test_transform)
        test_set = MyCIFAR100(file_path, train=False, download=True, transform=test_transform)
    elif args.dataset == 'ImageNet50':
        # Load Preprocessed IN-classes & indices; 50 classes were randomly selected
        index_path = args.data_path + '/ImageNet50/class_indice_dict.pickle'
        with open(index_path, 'rb') as f:
            class_indice_dict = pickle.load(f)
            #class_indice_dict['ood_indices'] = list(np.setdiff1d(list(range(0, len(train_set))), class_indice_dict['in_indices']))
        print(class_indice_dict.keys()) #['in_class', 'in_indices', 'in_indices_test', 'ood_indices']

        file_path = '/data/pdm102207/imagenet/'
        train_set = MyImageNet(file_path+'train/', transform=train_transform)
        unlabeled_set = MyImageNet(file_path + 'train/', transform=test_transform)
        test_set = MyImageNet(file_path+ 'val/', transform=test_transform)


    # class-split
    if args.dataset == 'CIFAR10':
        args.input_size = 32 * 32 * 3
        args.target_lists = [[4, 2, 5, 7], [7, 1, 2, 5], [6, 4, 3, 2], [8, 9, 1, 3], [2, 9, 5, 3]] # Randomly Selected
        args.target_list = args.target_lists[trial]
        args.untarget_list = list(np.setdiff1d(list(range(0, 10)), list(args.target_list)))
        args.num_IN_class = 4
    elif args.dataset == 'CIFAR100':
        args.input_size = 32 * 32 * 3
        args.target_lists = [[69,  8, 86, 18, 68, 30, 75,  3, 63, 76, 72,  7, 50, 81, 46, 89, 22,
            93, 62, 21, 33, 98, 82, 20, 60,  5, 77,  1, 74, 88, 57, 34, 43, 27, 66, 83, 25, 48,  4, 55], \
                            [33, 10, 74, 72, 88, 47, 27, 68, 60, 75, 45, 79, 92, 35, 86, 50, 18,
            61, 49, 29, 23, 30, 67, 73, 82, 94, 13, 37, 39, 26, 62, 22, 90, 53, 89, 11,  3, 20, 70, 96], \
                            [70, 28, 60, 22, 39, 35, 73, 13, 74, 10,  2, 16, 80, 53, 67, 66, 78,
            46, 26, 71, 43, 38, 42, 14, 50, 77, 20, 48, 52,  8, 54, 58, 91,  5, 25, 90, 61, 11, 59, 55], \
                            [ 7, 93, 37, 84, 57, 99, 10, 75, 54, 42, 26, 27, 47, 52, 61, 86, 60,
            90,  1,  0, 98, 87, 94, 74, 56, 91, 23, 97, 30, 17, 53, 12, 76, 11, 25, 65, 96,  3, 45, 8], \
                            [ 0,  1,  4,  5,  7,  9, 12, 19, 21, 22, 23, 24, 38, 41, 42, 43, 46,
            47, 48, 51, 55, 59, 60, 62, 68, 73, 75, 78, 79, 80, 81, 85, 86, 90,91, 94, 95, 96, 97, 98]] # Randomly Selected
        args.target_list = args.target_lists[trial]
        args.untarget_list = list(np.setdiff1d(list(range(0, 100)), list(args.target_list)))
        args.num_IN_class = 40
    elif args.dataset == 'ImageNet50':
        args.input_size = 32 * 32 * 3
        args.target_list = class_indice_dict['in_class']# SEED 1
        args.in_indices = class_indice_dict['in_indices']
        args.ood_indices = random.sample(list(np.setdiff1d(list(range(0, len(train_set))), list(args.in_indices))),
                                         round(1.5*len(args.in_indices)))
        #args.ood_indices = class_indice_dict['ood_indices']
        args.in_test_indices = class_indice_dict['in_indices_test']
        args.untarget_list = list(np.setdiff1d(list(range(0, 1000)), list(args.target_list)))
        args.num_IN_class = 50

    # class converting
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        for i, c in enumerate(args.untarget_list):
            train_set.targets[np.where(train_set.targets == c)[0]] = int(args.n_class)
            test_set.targets[np.where(test_set.targets == c)[0]] = int(args.n_class)

        args.target_list.sort()
        for i, c in enumerate(args.target_list):
            train_set.targets[np.where(train_set.targets == c)[0]] = i
            test_set.targets[np.where(test_set.targets == c)[0]] = i

        train_set.targets[np.where(train_set.targets == int(args.n_class))[0]] = int(args.num_IN_class)
        test_set.targets[np.where(test_set.targets == int(args.n_class))[0]] = int(args.num_IN_class)

    elif args.dataset == 'ImageNet50':
        args.target_list.sort()
        class_covert_dict = {}
        for i, c in enumerate(args.target_list):
            class_covert_dict[c] = i # {ori : split}

        for i, idx in enumerate(args.in_indices):
            train_set.targets[idx] = class_covert_dict[train_set.targets[idx]]

        for i, idx in enumerate(args.ood_indices):
            train_set.targets[idx] = int(args.num_IN_class)

        for i, idx in enumerate(args.in_test_indices):
            test_set.targets[idx] = class_covert_dict[test_set.targets[idx]]

    unlabeled_set.targets = train_set.targets

    # Split Check
    print("Target classes: ", args.target_list)
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        uni, cnt = np.unique(np.array(unlabeled_set.targets), return_counts=True)
        print("Train, # samples per class")
        print(uni, cnt)
        uni, cnt = np.unique(np.array(test_set.targets), return_counts=True)
        print("Test, # samples per class")
        print(uni, cnt)
    elif args.dataset == 'ImageNet50':
        uni, cnt = np.unique(np.array(unlabeled_set.targets[args.in_indices]), return_counts=True)
        print("Train IN, # samples per class")
        print(uni, cnt)
        uni, cnt = np.unique(np.array(unlabeled_set.targets[args.ood_indices]), return_counts=True)
        print("Train OOD (Sampled), # samples per class")
        print(uni, cnt)
        uni, cnt = np.unique(np.array(test_set.targets[args.in_test_indices]), return_counts=True)
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

def get_sub_train_dataset(args, dataset, L_index, O_index, U_index, Q_index, initial= False):
    classes = args.target_list
    budget = args.n_query
    ood_rate = args.ood_rate

    if initial:
        if args.dataset in ['CIFAR10', 'CIFAR100']:
            L_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
            O_total = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] >= len(classes)]

            n_ood = round(len(L_total) * (ood_rate / (1 - ood_rate)))
            O_total = random.sample(O_total, n_ood)
            print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))

            L_index = random.sample(L_total, int(budget * (1 - ood_rate)))
            O_index = random.sample(O_total, int(budget * ood_rate))
            U_index = list(set(L_total + O_total) - set(L_index) - set(O_index))
            print("# Labeled in: {}, ood: {}, Unlabeled: {}".format(len(L_index), len(O_index), len(U_index)))
        elif args.dataset == 'ImageNet50':
            # TODO: long time takes
            if initial:
                L_total = [dataset[i][2] for i in args.in_indices]
                O_total = [dataset[i][2] for i in args.ood_indices]

                n_ood = round(len(L_total) * (ood_rate / (1 - ood_rate)))
                O_total = random.sample(O_total, n_ood)
                print("# Total in: {}, ood: {}".format(len(L_total), len(O_total)))

                L_index = random.sample(L_total, int(budget * (1 - ood_rate)))
                O_index = random.sample(O_total, int(budget * ood_rate))
                U_index = list(set(L_total + O_total) - set(L_index) - set(O_index))
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

def get_sub_test_dataset(args, dataset):
    classes = args.target_list
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        labeled_index = [dataset[i][2] for i in range(len(dataset)) if dataset[i][1] < len(classes)]
    elif args.dataset == 'ImageNet50':
        labeled_index = [dataset[i][2] for i in args.in_test_indices]
    return labeled_index