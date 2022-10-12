from .almethod import ALMethod
import torch
import numpy as np

class Uncertainty(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, selection_method="CONF", **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)
        selection_choices = ["CONF", "Entropy", "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

    def run(self):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[:self.args.n_query]
        return selection_result, scores

    def rank_uncertainty(self):
        self.models['backbone'].eval()
        with torch.no_grad():
            selection_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size, num_workers=self.args.workers)

            scores = np.array([])
            batch_num = len(selection_loader)
            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data[0].to(self.args.device)
                if i % self.args.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "CONF":
                    preds, _ = self.models['backbone'](inputs)
                    confs = preds.max(axis=1).values.cpu().numpy()
                    scores = np.append(scores, confs)
                elif self.selection_method == "Entropy":
                    preds, _ = self.models['backbone'](inputs)
                    preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                    entropys = (np.log(preds + 1e-6) * preds).sum(axis=1)
                    scores = np.append(scores, entropys)
                elif self.selection_method == 'Margin':
                    preds, _ = self.models['backbone'](inputs)
                    preds = torch.nn.functional.softmax(preds, dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    margins = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()
                    scores = np.append(scores, margins)
        return scores

    def select(self, **kwargs):
        selected_indices, scores = self.run()
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores