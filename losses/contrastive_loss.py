import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from losses.gather import GatherLayer


class ContrastiveLoss(nn.Module):
    """ Supervised Contrastive Learning Loss among sample pairs.

    Args:
        scale (float): scaling factor.
    """
    def __init__(self, scale=16, **kwargs):
        super().__init__()
        self.s = scale

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normalize
        inputs = F.normalize(inputs, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)

        # compute cosine similarity
        similarities = torch.matmul(inputs, gallery_inputs.t()) * self.s
        
        # get mask for pos/neg pairs
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)

        loss = - loss.mean()

        return loss