from .almethod import ALMethod
import torch
import numpy as np
from .methods_utils.mqnet_util import *

class MQNet(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, cur_cycle, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        self.cur_cycle = cur_cycle

        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

    def select(self, **kwargs):
        unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)

        labeled_in_loader = torch.utils.data.DataLoader(self.labeled_in_set, shuffle=False, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, shuffle=False, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
        features_in = get_labeled_features(self.args, self.models, labeled_in_loader)

        if self.args.mqnet_mode == 'CONF':
            informativeness, features_unlabeled, _, _ = get_unlabeled_features(self.args, self.models, unlabeled_loader)
        if self.args.mqnet_mode == 'LL':
            informativeness, features_unlabeled, _, _ = get_unlabeled_features_LL(self.args, self.models, unlabeled_loader)

        purity = get_CSI_score(self.args, features_in, features_unlabeled)
        assert len(informativeness) == len(purity)

        if self.cur_cycle == 0:# initial round, MQNet is not trained yet
            if self.args.mqnet_mode == 'LL':
                informativeness, _, _ = standardize(informativeness)
            purity, _, _ = standardize(purity)
            query_scores = informativeness + purity
        else:
            meta_input = construct_meta_input(informativeness, purity)
            query_scores = self.models['mqnet'](meta_input)
        assert len(query_scores) == len(self.U_index_sub) #self.U_index

        selected_indices = np.argsort(-query_scores.reshape(-1).detach().cpu().numpy())[:self.args.n_query]
        Q_index = self.U_index_sub[selected_indices]

        #Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, query_scores