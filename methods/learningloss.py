from .almethod import ALMethod
import torch
import numpy as np

class LL(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, I_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        self.I_index = I_index
        self.labeled_in_set = torch.utils.data.Subset(self.unlabeled_dst, self.I_index)

    def run(self):
        # subset selection (for diversity)
        subset_idx = np.random.choice(len(self.U_index), size=(min(self.args.subset, len(self.U_index)),), replace=False)
        self.U_index_sub = np.array(self.U_index)[subset_idx]

        scores = self.get_predloss()
        selection_result = np.argsort(-scores)[:self.args.n_query]
        return selection_result, scores

    def get_predloss(self):
        self.models['backbone'].eval()
        self.models['module'].eval()
        loss_scores = torch.tensor([]).to(self.args.device)

        unlabeled_subset = torch.utils.data.Subset(self.unlabeled_dst, self.U_index_sub)
        with torch.no_grad():
            unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=self.args.test_batch_size, num_workers=self.args.workers)
            for data in unlabeled_loader:
                inputs = data[0].to(self.args.device)

                scores, features = self.models['backbone'](inputs)
                pred_loss = self.models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))

                loss_scores = torch.cat((loss_scores, pred_loss), 0)
        return loss_scores.cpu()

    def select(self, **kwargs):
        selected_idx_sub, scores = self.run()
        Q_index = self.U_index_sub[selected_idx_sub]

        return Q_index, scores