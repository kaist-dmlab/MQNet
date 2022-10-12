import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .transform_layers import *
from .ccal_util import set_random_seed, normalize
import datetime

def eval_unlabeled_detection(P, models,unlabeled_loader, train_loader,label_i_loader,simclr_aug=None):
    #base_path = os.path.split(P.load_path)[0]  # checkpoint directory
    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    #prefix = os.path.join(base_path, f'feats_{prefix}')
    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,  # 10
        'layers': ['simclr', 'shift']
    }

    print("----------------Get unlabeled data's semantic and distinctive feature-------------")
    feats_u_distinctive, labels, feats_u_semantic, index = get_features(P, P.dataset, models['distinctive'], models['semantic'], unlabeled_loader, prefix=prefix, **kwargs)  # (N, T, d)

    print("-------------Get each labeled data loader's semantic and distinctive feature------")
    feats_labeled_i_distinctive = []
    for i in range(len(label_i_loader)):
        feats_l_i_distinctive, label_l_i, feats_l_i_semantic, index_l_i = get_features(P, f'{P.dataset}_train', models['distinctive'], models['semantic'],label_i_loader[i], prefix=prefix,**kwargs)  # (M, T, d)
        feats_labeled_i_distinctive.append(feats_l_i_distinctive)

    print("------------------Get labeled data's semantic and distinctive feature-------------")
    feats_l_semantic, label_l, feats_l_semantic, index_l = get_features(P, f'{P.dataset}_train', models['distinctive'], models['semantic'], train_loader,prefix=prefix,**kwargs)  # (M, T, d)

    start = datetime.datetime.now()
    unlabeled_i_distinctive_score = []
    for i in range(len(feats_labeled_i_distinctive)):
        ood_score = 'CSI'
        P.axis = []
        for f in feats_labeled_i_distinctive[i]['simclr'].chunk(P.K_shift, dim=1):
            axis = f.mean(dim=1)
            P.axis.append(axis.to(P.device))
        #print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

        f_sim = [f.mean(dim=1) for f in feats_labeled_i_distinctive[i]['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
        f_shi = [f.mean(dim=1) for f in feats_labeled_i_distinctive[i]['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

        weight_sim = []
        weight_shi = []
        for shi in range(P.K_shift):
            sim_norm = f_sim[shi].norm(dim=1)
            shi_mean = f_shi[shi][:, shi]
            weight_sim.append(1 / sim_norm.mean().item())
            weight_shi.append(1 / shi_mean.mean().item())

        if ood_score == 'simclr':
            P.weight_sim = [1, 0, 0, 0]
            P.weight_shi = [0, 0, 0, 0]
        elif ood_score == 'CSI':
            P.weight_sim = weight_sim
            P.weight_shi = weight_shi
        else:
            raise ValueError()

        scores_u_i = get_scores_distinctive(P, feats_u_distinctive).numpy()
        score_distinctive_i = (scores_u_i - scores_u_i.min()) / (scores_u_i.max() - scores_u_i.min())
        unlabeled_i_distinctive_score.append(score_distinctive_i)

    print('CSI_last')
    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    ood_score = 'simclr'
    P.axis = []
    for f in feats_l_semantic['simclr'].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)
        P.axis.append(normalize(axis, dim=1).to(P.device))
    #print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    f_sim = [f.mean(dim=1) for f in feats_l_semantic['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_l_semantic['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    for shi in range(P.K_shift):
        sim_norm = f_sim[shi].norm(dim=1)
        shi_mean = f_shi[shi][:, shi]
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())

    if ood_score == 'simclr':
        P.weight_sim = [1, 0, 0, 0]
        P.weight_shi = [0, 0, 0, 0]
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print('simclr')
    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    print('Pre-compute features...')
    max_semantic,labels_semantic = get_scores_semantic(P, feats_u_semantic, label_l)
    max_semantic_score = (max_semantic - max_semantic.min()) / (max_semantic.max() - max_semantic.min())

    #Calculate score
    score_ours = []
    score_distinctive_i_list = []
    score_semantic_i_list = []
    query_index = []
    subset_index = []
    semantic_score = (1 - np.exp(-P.k * (max_semantic_score - P.t))) / (1 + np.exp(-P.k * (max_semantic_score - P.t)))

    for i in range(len(unlabeled_i_distinctive_score)):
        score_distinctive_i = (1 - torch.tensor(unlabeled_i_distinctive_score[i])) #* if_labels_i
        score_i = (semantic_score + score_distinctive_i) #* if_labels_i
        #score_i[labels_semantic != P.target_list[i]] = -1
        score_ours.append(score_i)
        score_distinctive_i_list.append(score_distinctive_i)
        score_semantic_i_list.append(semantic_score)
        query_index_i, query_label_i, query_score_i, semantic_i, distinctive_i, subset_index_i = Select_selector(
            index, labels, score_i, torch.tensor(max_semantic_score), score_distinctive_i, P)

        cnt = 0
        for j in range(len(subset_index_i)):
            if subset_index_i[j] not in subset_index:
                query_index += [query_index_i[j]]
                subset_index += [subset_index_i[j]]
                cnt+=1
            if cnt >= int(P.n_query)/P.target_number or len(subset_index) >= P.n_query:
                break

    end = datetime.datetime.now()
    print("time of calculate score:" + str((end - start).seconds) + "seconds")

    return query_index, subset_index


def Select_selector(select_indices,select_label, score,score_semantic,score_distinctive,args):

    score = torch.tensor(score)
    finally_selector, query_inside = torch.topk(score, len(score)) #int(args.n_query)/args.n_class
    query_inside = query_inside.cpu().data
    finally_label = np.asarray(select_label)[query_inside]
    finally_indices = np.asarray(select_indices)[query_inside]
    finally_semantic = np.asarray(score_semantic)[query_inside]
    finally_distinctive = np.asarray(score_distinctive)[query_inside]

    return finally_indices, finally_label, finally_selector, finally_semantic, finally_distinctive, np.asarray(query_inside)


def get_scores_distinctive(P, feats_dict):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(P.device)#[1600, 40, 128]
    feats_shi = feats_dict['shift'].to(P.device)#[1600, 40, 4]
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim1, f_shi in zip(feats_sim, feats_shi):#
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim1.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        f_sim_norm = [(f.mean(dim=0, keepdim=True)).norm(dim=1) for f in f_sim1.chunk(P.K_shift)]  # list of (1, d)

        score = 0
        for shi in range(P.K_shift):
            P_norm = P.axis[shi].norm(dim=1)
            cos_score = ((f_sim[shi] * P.axis[shi]).sum(dim=1)) / (f_sim_norm[shi].clone().detach() * P_norm.clone().detach())
            value, indices_e = cos_score.sort(descending=True)  # 降序
            score += torch.tensor(value[0].item()) - torch.tensor(value[1].item())
            anchor_A = (P.axis[shi][indices_e[0].item()]).view(1, -1)
            anchor_B = (P.axis[shi][indices_e[1].item()]).view(1, -1)
            cos_score_AB = ((anchor_A * anchor_B).sum(dim=1)) / (anchor_A.norm(dim=1).clone().detach() * anchor_B.norm(dim=1).clone().detach())

            score += torch.tensor(cos_score_AB.item()) + torch.tensor(value[0].item())
        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_scores_semantic(P, feats_dict,labels):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(P.device)#[1600, 40, 128]
    feats_shi = feats_dict['shift'].to(P.device)#[1600, 40, 4]
    N = feats_sim.size(0)

    # compute scores
    maxs = []
    labels_semantic = []

    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [normalize(f.mean(dim=0, keepdim=True), dim=1) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)

        max_simi = 0
        for shi in range(P.K_shift):
            value_sim, indices_sim = ((f_sim[shi] * P.axis[shi]).sum(dim=1)).sort(descending=True)
            max_simi += value_sim.max().item() * P.weight_sim[shi]

            if shi == 0:
                labels_semantic.append(labels[indices_sim[0].item()])
        maxs.append(max_simi)
    maxs = torch.tensor(maxs)
    labels_semantic = torch.tensor(labels_semantic)

    assert maxs.dim() == 1 and maxs.size(0) == N  # (N)
    return maxs.cpu(),labels_semantic.cpu()


def get_features(P, data_name, model_distinctive, model_semantic,loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    feats_dict_distinctive = dict()
    feats_dict_semantic = dict()
    left = [layer for layer in layers if layer not in feats_dict_distinctive.keys()]

    if len(left) > 0:
        _feats_dict_distinctive,labels,_feats_dict_semantic,index = _get_features(P, model_distinctive,model_semantic, loader, interp, P.dataset == 'imagenet',simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict_distinctive.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            #torch.save(_feats_dict_distinctive[layer], path)
            feats_dict_distinctive[layer] = feats  # update value
        for layer, feats in _feats_dict_semantic.items():
            path = prefix + '2' + f'_{data_name}_{layer}.pth'
            #torch.save(_feats_dict_semantic[layer], path)
            feats_dict_semantic[layer] = feats  # update value

    return feats_dict_distinctive,labels,feats_dict_semantic,index


def _get_features(P, model_distinctive,model_semantic, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    hflip = HorizontalFlipLayer().to(P.device)

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    labels = []
    index = []
    model_distinctive.eval()
    model_semantic.eval()
    feats_all_distinctive = {layer: [] for layer in layers}  # initialize: empty list
    feats_all_semantic = {layer: [] for layer in layers}
    for i, (x, label, indices) in enumerate(loader):
        labels.extend(label)
        index.extend(indices)
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(P.device)  # gpu tensor

        # compute features in one batch
        feats_batch_distinctive = {layer: [] for layer in layers}  # initialize: empty list
        feats_batch_semantic = {layer: [] for layer in layers}

        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux_distinctive = model_distinctive(x_t, **kwargs)
                _, output_aux_semantic = model_semantic(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats_distinctive = output_aux_distinctive[layer].cpu()
                feats_semantic = output_aux_semantic[layer].cpu()
                if imagenet is False:
                    feats_batch_distinctive[layer] += feats_distinctive.chunk(P.K_shift)
                    feats_batch_semantic[layer] += feats_semantic.chunk(P.K_shift)
                else:
                    feats_batch_distinctive[layer] += [feats_distinctive]  # (B, d) cpu tensor
                    feats_batch_semantic[layer] += [feats_semantic]

        # concatenate features in one batch
        for key, val in feats_batch_distinctive.items():
            if imagenet:
                feats_batch_distinctive[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch_distinctive[key] = torch.stack(val, dim=1)  # (B, T, d)

        for key, val in feats_batch_semantic.items():
            if imagenet:
                feats_batch_semantic[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch_semantic[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all_distinctive[layer] += [feats_batch_distinctive[layer]]
            feats_all_semantic[layer] += [feats_batch_semantic[layer]]

    # concatenate features in full dataset
    for key, val in feats_all_distinctive.items():
        feats_all_distinctive[key] = torch.cat(val, dim=0)  # (N, T, d)
    for key, val in feats_all_semantic.items():
        feats_all_semantic[key] = torch.cat(val, dim=0)  # (N, T, d)


    # reshape order
    if imagenet is False:
        for key, val in feats_all_distinctive.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all_distinctive[key] = val
    if imagenet is False:
        for key, val in feats_all_semantic.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all_semantic[key] = val

    return feats_all_distinctive,labels,feats_all_semantic,index

'''
def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))


def test_classifier(P, model, loader, steps, marginal=False, logger=None):
    error_top1 = AverageMeter()
    error_calibration = AverageMeter()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # Switch to evaluate mode
    mode = model.training
    model.eval()

    for n, (images, labels,index) in enumerate(loader):
        batch_size = images.size(0)

        images, labels = images.to(P.device), labels.to(P.device)

        if marginal:
            outputs = 0
            for i in range(4):
                rot_images = torch.rot90(images, i, (2, 3))
                _, outputs_aux = model(rot_images, joint=True)
                outputs += outputs_aux['joint'][:, P.n_classes * i: P.n_classes * (i + 1)] / 4.
        else:
            outputs = model(images)

        top1, = error_k(outputs.data, labels, ks=(1,))
        error_top1.update(top1.item(), batch_size)

        ece = ece_criterion(outputs, labels) * 100
        error_calibration.update(ece.item(), batch_size)

        if n % 100 == 0:
            log_('[Test %3d] [Test@1 %.3f] [ECE %.3f]' %
                 (n, error_top1.value, error_calibration.value))

    log_(' * [Error@1 %.3f] [ECE %.3f]' %
         (error_top1.average, error_calibration.average))

    if logger is not None:
        logger.scalar_summary('eval/clean_error', error_top1.average, steps)
        logger.scalar_summary('eval/ece', error_calibration.average, steps)

    model.train(mode)

    return error_top1.average
'''