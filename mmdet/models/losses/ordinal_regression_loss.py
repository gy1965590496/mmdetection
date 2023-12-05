import torch
import torch.nn as nn

class OccOrdinalRegressionLoss(nn.Module):
    def __init__(self, num_cls=5, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_cls = num_cls
        self.cutpoints = torch.arange(num_cls-1).float()*scale/(num_cls-2) - scale / 2
        # self.cutpoints = torch.arange(0 + 1.0 / num_class, 1.0, 1.0 / num_class)
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def ord_reg_likelihoods(self, pred): 
        """
            pred: torch.Size([n, 1])
        """
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(-1)       
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)
        return likelihoods

    def forward(self, pred, label):
        """
        pred:5*1
        label:5*1
        """
        # 自动广播
        # self.cutpoints -> torch.size([num_class-1]) -> 广播为 torch.size([num_class, num_class-1])
        # pred -> torch.size([num_class, 1])
        # self.cutpoints - pred -> torch.size([num_class, num_class-1])
        # 其含义为，pred分别和4个分界点的差异
        assert pred.shape[0] == label.shape[0]
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, 1, label).mean()
            
        return loss, likelihoods