import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as dist
from losses.gather import GatherLayer


class CosFaceLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        output = self.s * (inputs - one_hot * self.m)

        return F.cross_entropy(output, targets)


class PairwiseCosFaceLoss(nn.Module):
    """ CosFace Loss among sample pairs.

    Reference:
        Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0):
        super().__init__()
        self.s = scale
        self.m = margin

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
        similarities = torch.matmul(inputs, gallery_inputs.t())
        
        # get mask for pos/neg pairs
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        scores = (similarities + self.m) * mask_neg - similarities * mask_pos
        scores = scores * self.s
        
        neg_scores_LSE = torch.logsumexp(scores * mask_neg - 99999999 * (1 - mask_neg), dim=1)
        pos_scores_LSE = torch.logsumexp(scores * mask_pos - 99999999 * (1 - mask_pos), dim=1)

        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()

        return loss