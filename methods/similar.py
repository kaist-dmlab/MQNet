from .almethod import ALMethod
import torch
import numpy as np

from .methods_utils.scmi import SCMI # need to prior-install

class SIMILAR(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, O_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.O_index = O_index

    def get_SCMI_indice(self):
        # Define arguments for SCMI
        selection_strategy_args = {'device': self.args.device,
                                   'batch_size': self.args.batch_size,
                                   'scmi_function': self.args.submodular,  # flcmi, logdetcmi
                                   'metric': 'cosine', # Use cosine similarity when determining the likeness of two data points
                                   'optimizer': self.args.submodular_greedy, # When doing submodular maximization, use the lazy greedy optimizer
                                   'class_dict': None,
                                   'ood_name': None
                                   }
        unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)

        query_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)
        if self.args.ood_rate == 0:
            private_set = torch.utils.data.Subset(self.unlabeled_dst, [0])
        else:
            private_set = torch.utils.data.Subset(self.unlabeled_dst, self.O_index)

        selection_strategy = SCMI(self.unlabeled_dst, unlabeled_subset, query_set, private_set,
                                  self.models['backbone'], self.args.num_IN_class + 1, selection_strategy_args)

        # Do the selection, which will return the indices of the selected points with respect to the unlabeled dataset.
        selected_idx_sub = selection_strategy.select(self.args.n_query)

        return selected_idx_sub

    def select(self, **kwargs):
        # subset selection (if needed, to avoid out of memory)
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

        selected_idx_sub = self.get_SCMI_indice()
        scores = list(np.ones(len(selected_idx_sub)))  # equally assign 1 (meaningless)

        Q_index = self.U_index_sub[selected_idx_sub]

        return Q_index, scores