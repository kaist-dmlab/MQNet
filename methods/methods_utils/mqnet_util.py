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


def get_unlabeled_features(args, models, unlabeled_loader):
    models['csi'].eval()
    layers = ('simclr', 'shift')
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    kwargs = {layer: True for layer in layers}

    # generate entire unlabeled features set
    features_unlabeled = torch.tensor([]).to(args.device)
    uncertainty = torch.tensor([]).to(args.device)
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
        uncertainty = torch.cat((uncertainty, u), 0)

        indices = torch.cat((indices, index), 0)
    uncertainty = 1 - uncertainty

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

    return sim_scores.type(torch.float32).to(args.device).reshape((-1, 1))

def standardize(scores):
    std, mean = torch.std_mean(scores, unbiased=False)
    scores = (scores - mean) / std
    scores = torch.exp(scores)
    return scores

def construct_meta_input(informativeness, purity):
    informativeness = standardize(informativeness)
    purity = standardize(purity)

    meta_input = torch.cat((informativeness, purity), 1)
    return meta_input