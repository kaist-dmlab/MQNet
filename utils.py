from argparse import ArgumentTypeError
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
from torchlars import LARS
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Subset

from methods.methods_utils.mqnet_util import *
from methods.methods_utils.ccal_util import *
from methods.methods_utils.simclr import semantic_train_epoch
from methods.methods_utils.simclr_CSI import csi_train_epoch


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

def semantic_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train a Semantic Model.')
    time_start = time.time()

    for epoch in range(args.epochs_ccal):
        semantic_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def distinctive_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train a Distinctive Model.')
    time_start = time.time()

    for epoch in range(args.epochs_ccal):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def csi_train(args, model, criterion, optimizer, scheduler, loader, simclr_aug=None, linear=None, linear_optim=None):
    print('>> Train CSI.')
    time_start = time.time()

    for epoch in range(args.epochs_csi):
        csi_train_epoch(args, epoch, model, criterion, optimizer, scheduler, loader, simclr_aug, linear, linear_optim)
        scheduler.step()
    print('>> Finished, Elapsed Time: {}'.format(time.time()-time_start))

def self_sup_train(args, trial, models, optimizers, schedulers, train_dst, I_index, O_index, U_index):
    criterion = nn.CrossEntropyLoss()

    train_in_data = Subset(train_dst, I_index)
    train_ood_data = Subset(train_dst, O_index)
    train_unlabeled_data = Subset(train_dst, U_index)
    print("Self-sup training, # in: {}, # ood: {}, # unlabeled: {}".format(len(train_in_data), len(train_ood_data), len(train_unlabeled_data)))

    datalist = [train_in_data, train_ood_data, train_unlabeled_data]
    multi_datasets = torch.utils.data.ConcatDataset(datalist)

    if args.method == 'CCAL':
        # if a pre-trained CSI exist, just load it
        semantic_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_semantic_' + str(trial) + '.pt'
        distinctive_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_distinctive_' + str(trial) + '.pt'
        if os.path.isfile(semantic_path) and os.path.isfile(distinctive_path):
            print('Load pre-trained semantic, distinctive models, named: {}, {}'.format(semantic_path, distinctive_path))
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)
            models['semantic'].load_state_dict(torch.load(semantic_path))
            models['distinctive'].load_state_dict(torch.load(distinctive_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(dataset=multi_datasets, batch_size=args.ccal_batch_size, shuffle=True)
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training the Semantic Coder
            if args.data_parallel == True:
                linear = models['semantic'].module.linear
            else:
                linear = models['semantic'].linear
            linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
            args.shift_trans_type = 'none'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            semantic_train(args, models['semantic'], criterion, optimizers['semantic'], schedulers['semantic'],
                           contrastive_loader, simclr_aug, linear, linear_optim)

            # Training the Distinctive Coder
            if args.data_parallel == True:
                linear = models['distinctive'].module.linear
            else:
                linear = models['distinctive'].linear
            linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
            args.shift_trans_type = 'rotation'
            args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
            args.shift_trans = args.shift_trans.to(args.device)

            distinctive_train(args, models['distinctive'], criterion, optimizers['distinctive'], schedulers['distinctive'],
                              contrastive_loader, simclr_aug, linear, linear_optim)

            # SSL save
            if args.ssl_save == True:
                torch.save(models['semantic'].state_dict(), semantic_path)
                torch.save(models['distinctive'].state_dict(), distinctive_path)

    elif args.method == 'MQNet':
        if args.data_parallel == True:
            linear = models['csi'].module.linear
        else:
            linear = models['csi'].linear
        linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=args.weight_decay)
        args.shift_trans_type = 'rotation'
        args.shift_trans, args.K_shift = get_shift_module(args, eval=True)
        args.shift_trans = args.shift_trans.to(args.device)

        # if a pre-trained CSI exist, just load it
        model_path = 'weights/'+ str(args.dataset)+'_r'+str(args.ood_rate)+'_csi_'+str(trial) + '.pt'
        if os.path.isfile(model_path):
            print('Load pre-trained CSI model, named: {}'.format(model_path))
            models['csi'].load_state_dict(torch.load(model_path))
        else:
            contrastive_loader = torch.utils.data.DataLoader(dataset=multi_datasets, batch_size=args.csi_batch_size, shuffle=True)
            simclr_aug = get_simclr_augmentation(args, image_size=(32, 32, 3)).to(args.device)  # for CIFAR10, 100

            # Training CSI
            csi_train(args, models['csi'], criterion, optimizers['csi'], schedulers['csi'],
                      contrastive_loader, simclr_aug, linear, linear_optim)

            # SSL save
            if args.ssl_save == True:
                torch.save(models['csi'].state_dict(), model_path)

    return models

def mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict):
    models['mqnet'].train()
    models['backbone'].eval()

    batch_idx = 0
    while (batch_idx < args.steps_per_epoch):
        for data in delta_loader:
            optimizers['mqnet'].zero_grad()
            inputs, labels, indexs = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

            # get pred_scores through MQNet
            meta_inputs = torch.tensor([]).to(args.device)
            in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
            for idx in indexs:
                meta_inputs = torch.cat((meta_inputs, meta_input_dict[idx.item()][0].reshape((-1, 2))), 0)
                in_ood_masks = torch.cat((in_ood_masks, meta_input_dict[idx.item()][1]), 0)

            pred_scores = models['mqnet'](meta_inputs)

            # get target loss
            mask_labels = labels*in_ood_masks # make the label of OOD points to 0 (to calculate loss)

            out, features = models['backbone'](inputs)
            true_loss = criterion(out, mask_labels)  # ground truth loss
            mask_true_loss = true_loss*in_ood_masks # make the true_loss of OOD points to 0

            loss = LossPredLoss(pred_scores, mask_true_loss.reshape((-1, 1)), margin=1)

            loss.backward()
            optimizers['mqnet'].step()

            batch_idx += 1

def mqnet_train(args, models, optimizers, schedulers, criterion, delta_loader, meta_input_dict):
    print('>> Train MQNet.')
    for epoch in tqdm(range(args.epochs_mqnet), leave=False, total=args.epochs_mqnet):
        mqnet_train_epoch(args, models, optimizers, criterion, delta_loader, meta_input_dict)
        schedulers['mqnet'].step()
    print('>> Finished.')

def meta_train(args, models, optimizers, schedulers, criterion, labeled_in_loader, unlabeled_loader, delta_loader):
    features_in = get_labeled_features(args, models, labeled_in_loader)

    if args.mqnet_mode == 'CONF':
        informativeness, features_delta, in_ood_masks, indices = get_unlabeled_features(args, models, delta_loader)
    elif args.mqnet_mode == 'LL':
        informativeness, features_delta, in_ood_masks, indices = get_unlabeled_features_LL(args, models, delta_loader)

    purity = get_CSI_score(args, features_in, features_delta)
    assert informativeness.shape == purity.shape

    if args.mqnet_mode == 'CONF':
        meta_input = construct_meta_input(informativeness, purity)
    elif args.mqnet_mode == 'LL':
        meta_input = construct_meta_input_with_U(informativeness, purity, args, models, unlabeled_loader)

    # For enhancing training efficiency, generate meta-input & in-ood masks once, and save it into a dictionary
    meta_input_dict = {}
    for i, idx in enumerate(indices):
        meta_input_dict[idx.item()] = [meta_input[i].to(args.device), in_ood_masks[i]]

    # Mini-batch Training
    mqnet_train(args, models, optimizers, schedulers, criterion, delta_loader, meta_input_dict)

    return models

def train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders):
    models['backbone'].train()
    models['module'].train()

    batch_idx = 0
    while (batch_idx < args.steps_per_epoch):
        for data in dataloaders['train']:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            # Classification loss for in-distribution
            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

            # loss module for predLoss
            if epoch > args.epoch_loss:
                # After 120 epochs, stop the gradient from the loss prediction module
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss = LossPredLoss(pred_loss, target_loss, margin=1)

            loss = m_backbone_loss + m_module_loss
            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()

            batch_idx += 1

def train_epoch(args, models, criterion, optimizers, dataloaders):
    models['backbone'].train()

    batch_idx = 0
    while(batch_idx < args.steps_per_epoch):
        for data in dataloaders['train']:
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            optimizers['backbone'].zero_grad()

            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)

            loss = m_backbone_loss
            loss.backward()
            optimizers['backbone'].step()

            batch_idx+=1
            #if batch_idx >= steps_per_epoch:
            #    break

def train(args, models, criterion, optimizers, schedulers, dataloaders):
    print('>> Train a Model.')
    print("num_epochs: {}, steps_per_epoch: {}, total_update: {}".format(
            args.epochs, args.steps_per_epoch, int(args.epochs*args.steps_per_epoch)) )

    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'CCAL', 'SIMILAR']:
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch(args, models, criterion, optimizers, dataloaders)
            schedulers['backbone'].step()

    elif args.method in ['LL']: #MQNet
        for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
            train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders)
            schedulers['backbone'].step()
            schedulers['module'].step()

    elif args.method in ['MQNet']: #MQNet
        if args.mqnet_mode == "CONF":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                train_epoch(args, models, criterion, optimizers, dataloaders)
                schedulers['backbone'].step()
        elif args.mqnet_mode == "LL":
            for epoch in tqdm(range(args.epochs), leave=False, total=args.epochs):
                train_epoch_LL(args, models, epoch, criterion, optimizers, dataloaders)
                schedulers['backbone'].step()
                schedulers['module'].step()

    print('>> Finished.')

def test(args, models, dataloaders):
    top1 = AverageMeter('Acc@1', ':6.2f')

    # Switch to evaluate mode
    models['backbone'].eval()
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)

            # Compute output
            with torch.no_grad():
                scores, _ = models['backbone'](inputs)

            # Measure accuracy and record loss
            prec1 = accuracy(scores.data, labels, topk=(1,))[0]
            top1.update(prec1.item(), inputs.size(0))
        print('Test acc: * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def get_more_args(args):
    cuda = ""
    if len(args.gpu) > 1:
        cuda = 'cuda'
    elif len(args.gpu) == 1:
        cuda = 'cuda:' + str(args.gpu[0])

    if args.dataset == 'ImageNet':
        args.device = cuda if torch.cuda.is_available() else 'cpu'
    else:
        args.device = cuda if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'CIFAR10':
        args.channel = 3
        args.im_size = (32, 32)
        #args.num_IN_class = 4

    elif args.dataset == 'CIFAR100':
        args.channel = 3
        args.im_size = (32, 32)
        #args.num_IN_class = 40

    elif args.dataset == 'ImageNet50':
        args.channel = 3
        args.im_size = (224, 224)
        #args.num_IN_class = 50

    return args

def get_models(args, nets, model, models):
    # Normal
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE']:
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
        models = {'backbone': backbone}

    # SIMILAR
    elif args.method =='SIMILAR':
        backbone = nets.__dict__[model](args.channel, args.num_IN_class+1, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
        models = {'backbone': backbone}

    # LL
    elif args.method == 'LL':
        model_ = model + '_LL'
        backbone = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        loss_module = nets.__dict__['LossNet'](args.im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            loss_module = nets.nets_utils.MyDataParallel(loss_module, device_ids=args.gpu)

        models = {'backbone': backbone, 'module': loss_module}

    # CCAL
    elif args.method == 'CCAL':
        backbone = nets.__dict__[model](args.channel, args.num_IN_class, args.im_size).to(args.device)

        model_ = model+'_CSI'
        model_sem = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        model_dis = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            model_sem = nets.nets_utils.MyDataParallel(model_sem, device_ids=args.gpu)
            model_dis = nets.nets_utils.MyDataParallel(model_dis, device_ids=args.gpu)

        if models == None: #initial round
            models = {'backbone': backbone, 'semantic': model_sem, 'distinctive': model_dis}
        else:
            models['backbone'] = backbone

    # MQNet
    elif args.method == 'MQNet':
        model_ = model + '_LL'
        backbone = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)
        loss_module = nets.__dict__['LossNet'](args.im_size).to(args.device)

        model_ = model + '_CSI'
        model_csi = nets.__dict__[model_](args.channel, args.num_IN_class, args.im_size).to(args.device)

        if args.device == "cpu":
            print("Using CPU.")
        elif args.data_parallel == True:
            backbone = nets.nets_utils.MyDataParallel(backbone, device_ids=args.gpu)
            loss_module = nets.nets_utils.MyDataParallel(loss_module, device_ids=args.gpu)
            model_csi = nets.nets_utils.MyDataParallel(model_csi, device_ids=args.gpu)

        if models == None: #initial round
            models = {'backbone': backbone, 'module': loss_module, 'csi': model_csi} #, 'mqnet': mqnet
        else:
            models['backbone'] = backbone
            models['module'] = loss_module

    return models

def init_mqnet(args, nets, models, optimizers, schedulers):
    models['mqnet'] = nets.__dict__['QueryNet'](input_size=2, inter_dim=64).to(args.device)

    optim_mqnet = torch.optim.SGD(models['mqnet'].parameters(), lr=args.lr_mqnet)
    sched_mqnet = torch.optim.lr_scheduler.MultiStepLR(optim_mqnet, milestones=[int(args.epochs_mqnet / 2)])

    optimizers['mqnet'] = optim_mqnet
    schedulers['mqnet'] = sched_mqnet
    return models, optimizers, schedulers

def get_optim_configurations(args, models):
    print("lr: {}, momentum: {}, decay: {}".format(args.lr, args.momentum, args.weight_decay))
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(models['backbone'].parameters(), args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(models['backbone'].parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.__dict__[args.optimizer](models['backbone'].parameters(), args.lr, momentum=args.momentum,
                                                         weight_decay=args.weight_decay)

    # LR scheduler
    if args.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    elif args.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone)
    else:
        scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer)

    # Normal
    if args.method in ['Random', 'Uncertainty', 'Coreset', 'BADGE', 'SIMILAR']:
        optimizers = {'backbone': optimizer}
        schedulers = {'backbone': scheduler}

    # LL (+ loss_pred module)
    elif args.method == 'LL':
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=args.milestone)

        optimizers = {'backbone': optimizer, 'module': optim_module}
        schedulers = {'backbone': scheduler, 'module': sched_module}

    # CCAL (+ 2 contrastive coders)
    elif args.method == 'CCAL':
        optim_sem = torch.optim.SGD(models['semantic'].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_sem = torch.optim.lr_scheduler.CosineAnnealingLR(optim_sem, args.epochs_ccal, eta_min=args.min_lr)
        scheduler_warmup_sem = GradualWarmupScheduler(optim_sem, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_sem)

        optim_dis = torch.optim.SGD(models['distinctive'].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_dis = torch.optim.lr_scheduler.CosineAnnealingLR(optim_dis, args.epochs_ccal, eta_min=args.min_lr)
        scheduler_warmup_dis = GradualWarmupScheduler(optim_dis, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_dis)

        optimizers = {'backbone': optimizer, 'semantic': optim_sem, 'distinctive': optim_dis}
        schedulers = {'backbone': scheduler, 'semantic': scheduler_warmup_sem, 'distinctive': scheduler_warmup_dis}

    # MQ-Net
    elif args.method == 'MQNet':
        optim_module = torch.optim.SGD(models['module'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sched_module = torch.optim.lr_scheduler.MultiStepLR(optim_module, milestones=args.milestone)

        optimizer_csi = torch.optim.SGD(models['csi'].parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
        optim_csi = LARS(optimizer_csi, eps=1e-8, trust_coef=0.001)

        sched_csi = torch.optim.lr_scheduler.CosineAnnealingLR(optim_csi, args.epochs_csi)
        scheduler_warmup_csi = GradualWarmupScheduler(optim_csi, multiplier=10.0, total_epoch=args.warmup, after_scheduler=sched_csi)

        optimizers = {'backbone': optimizer, 'module': optim_module, 'csi': optim_csi}
        schedulers = {'backbone': scheduler, 'module': sched_module, 'csi': scheduler_warmup_csi}

    return criterion, optimizers, schedulers