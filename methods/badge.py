from .almethod import ALMethod
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats

class BADGE(ALMethod):
    def __init__(self, args, models, unlabeled_dst, U_index, **kwargs):
        super().__init__(args, models, unlabeled_dst, U_index, **kwargs)

    def get_grad_features(self):
        self.models['backbone'].eval()
        embDim = self.models['backbone'].get_embedding_dim() # 512?
        num_unlabeled = len(self.U_index)
        n_class = self.args.num_IN_class
        grad_embeddings = torch.zeros([num_unlabeled, embDim * n_class])
        with torch.no_grad():
            unlabeled_loader = torch.utils.data.DataLoader(self.unlabeled_set, batch_size=self.args.test_batch_size,
                                                           num_workers=self.args.workers)
            # generate entire unlabeled features set
            for i, (inputs, labels, _) in enumerate(unlabeled_loader):
                inputs = inputs.to(self.args.device)
                out, features = self.models['backbone'](inputs)
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                maxInds = torch.argmax(batchProbs, 1)
                # _, preds = torch.max(out.data, 1)

                for j in range(len(inputs)):
                    for c in range(n_class):
                        if c == maxInds[j]:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        1 - batchProbs[j][c])
                        else:
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                        -1 * batchProbs[j][c])
        return grad_embeddings.cpu().numpy()

    # kmeans ++ initialization
    def k_means_plus_centers(self, X, K):
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        mu = [X[ind]]
        indsAll = [ind]
        centInds = [0.] * len(X)
        cent = 0
        print('#Samps\tTotal Distance')
        while len(mu) < K:
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            if len(mu) % 100 == 0:
                print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            if sum(D2) == 0.0: pdb.set_trace()
            D2 = D2.ravel().astype(float)
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        return indsAll

    def select(self, **kwargs):
        unlabeled_features = self.get_grad_features()
        selected_indices = self.k_means_plus_centers(X=unlabeled_features, K=self.args.n_query)
        scores = list(np.ones(len(selected_indices))) # equally assign 1 (meaningless)

        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index, scores