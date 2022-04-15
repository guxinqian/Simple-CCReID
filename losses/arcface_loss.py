import math
import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceLoss(nn.Module):
    """ ArcFace loss.

    Reference:
        Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In CVPR, 2019.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """
    def __init__(self, scale=16, margin=0.1):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        # get a one-hot index
        index = inputs.data * 0.0 
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = inputs[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m  - sin_t * sin_m

        cond_v = cos_t - math.cos(math.pi - self.m)
        cond = F.relu(cond_v)
        keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = inputs * 1.0 
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, targets)
