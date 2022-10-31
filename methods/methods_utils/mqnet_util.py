import torch
import torch.nn as nn
import time
import numpy as np
import random

def get_labeled_features(args, models, labeled_in_loader):
    models['csi'].eval()

    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    features_in = torch.tensor([]).to(args.device)
    for data in labeled_in_loader:
        inputs = data[0].to(args.device)
        _, couts = models['csi'](inputs, **kwargs)
        features_in = torch.cat((features_in, couts['simclr'].detach()), 0)
    return features_in

def get_unlabeled_features_LL_with_U(args, models, delta_loader, unlabeled_loader):
    models['csi'].eval()
    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    # generate entire unlabeled features set
    features_unlabeled = torch.tensor([]).to(args.device)
    pred_loss = torch.tensor([]).to(args.device)
    in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
    indices = torch.tensor([]).type(torch.LongTensor).to(args.device)

    for data in delta_loader:
        inputs = data[0].to(args.device)
        labels = data[1].to(args.device)
        index = data[2].to(args.device)

        in_ood_mask = labels.le(args.num_IN_class - 1).type(torch.LongTensor).to(args.device)
        in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

        out, couts = models['csi'](inputs, **kwargs)
        features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

        out, features = models['backbone'](inputs)
        pred_l = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
        pred_l = pred_l.view(pred_l.size(0))
        pred_loss = torch.cat((pred_loss, pred_l.detach()), 0)

        indices = torch.cat((indices, index), 0)

    return pred_loss.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices

def get_unlabeled_features_LL(args, models, unlabeled_loader):
    models['csi'].eval()
    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    # generate entire unlabeled features set
    features_unlabeled = torch.tensor([]).to(args.device)
    pred_loss = torch.tensor([]).to(args.device)
    in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
    indices = torch.tensor([]).type(torch.LongTensor).to(args.device)

    for data in unlabeled_loader:
        inputs = data[0].to(args.device)
        labels = data[1].to(args.device)
        index = data[2].to(args.device)

        in_ood_mask = labels.le(args.num_IN_class - 1).type(torch.LongTensor).to(args.device)
        in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

        out, couts = models['csi'](inputs, **kwargs)
        features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

        out, features = models['backbone'](inputs)
        pred_l = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
        pred_l = pred_l.view(pred_l.size(0))
        pred_loss = torch.cat((pred_loss, pred_l.detach()), 0)

        indices = torch.cat((indices, index), 0)

    return pred_loss.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices

def get_unlabeled_features(args, models, unlabeled_loader):
    models['csi'].eval()
    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    # generate entire unlabeled features set
    features_unlabeled = torch.tensor([]).to(args.device)
    conf = torch.tensor([]).to(args.device)
    in_ood_masks = torch.tensor([]).type(torch.LongTensor).to(args.device)
    indices = torch.tensor([]).type(torch.LongTensor).to(args.device)

    f = nn.Softmax(dim=1)
    for data in unlabeled_loader:
        inputs = data[0].to(args.device)
        labels = data[1].to(args.device)
        index = data[2].to(args.device)

        in_ood_mask = labels.le(args.num_IN_class-1).type(torch.LongTensor).to(args.device)
        in_ood_masks = torch.cat((in_ood_masks, in_ood_mask.detach()), 0)

        out, couts = models['csi'](inputs, **kwargs)
        features_unlabeled = torch.cat((features_unlabeled, couts['simclr'].detach()), 0)

        out, features = models['backbone'](inputs)
        u, _ = torch.max(f(out.data), 1)
        conf = torch.cat((conf, u), 0)

        indices = torch.cat((indices, index), 0)
    uncertainty = 1-conf

    return uncertainty.reshape((-1, 1)), features_unlabeled, in_ood_masks.reshape((-1, 1)), indices


def get_CSI_score(args, features_in, features_unlabeled):
    # CSI Score
    sim_scores = torch.tensor([]).to(args.device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    for f_u in features_unlabeled:
        f_u_expand = f_u.reshape((1, -1)).expand((len(features_in), -1))
        sim = cos(f_u_expand, features_in)  # .reshape(-1,1)
        max_sim, _ = torch.max(sim, 0)
        # score = max_sim * torch.norm(f_u)
        sim_scores = torch.cat((sim_scores, max_sim.reshape(1)), 0)

    # similarity = negative distance = nagative OODness
    return sim_scores.type(torch.float32).to(args.device).reshape((-1, 1))

def standardize(scores):
    std, mean = torch.std_mean(scores, unbiased=False)
    scores = (scores - mean) / std
    scores = torch.exp(scores)
    return scores, std, mean

def standardize_with_U(scores, scores_U):
    std, mean = torch.std_mean(scores_U, unbiased=False)
    scores = (scores - mean) / std
    scores = torch.exp(scores)
    return scores, std, mean

def construct_meta_input(informativeness, purity):
    informativeness, std, mean = standardize(informativeness)
    print("informativeness mean: {}, std: {}".format(mean, std))

    purity, std, mean = standardize(purity)
    print("purity mean: {}, std: {}".format(mean, std))

    # TODO:
    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input

def construct_meta_input_with_U(informativeness, purity, args, models, unlabeled_loader):
    informativeness_U, _, _, _ = get_unlabeled_features(args, models, unlabeled_loader)

    scores, std, mean = standardize_with_U(informativeness, informativeness_U)
    print("informativeness mean: {}, std: {}".format(mean, std))

    purity, std, mean = standardize(purity)
    print("purity mean: {}, std: {}".format(mean, std))

    # TODO:
    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input