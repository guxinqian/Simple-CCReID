import math
import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer


class TripletLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normlize
        inputs = F.normalize(inputs, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)

        # compute distance
        dist = 1 - torch.matmul(inputs, gallery_inputs.t()) # values in [0, 2]

        # get positive and negative masks
        targets, gallery_targets = targets.view(-1,1), gallery_targets.view(-1,1)
        mask_pos = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_neg = 1 - mask_pos

        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.max((dist - mask_neg * 99999999.), dim=1)
        dist_an, _ = torch.min((dist + mask_pos * 99999999.), dim=1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss