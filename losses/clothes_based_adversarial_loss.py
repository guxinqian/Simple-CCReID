import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer


class ClothesBasedAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg

        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()

        return loss


class ClothesBasedAdversarialLossWithMemoryBank(nn.Module):
    """ Clothes-based Adversarial Loss between mini batch and the samples in memory.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        num_clothes (int): the number of clothes classes.
        feat_dim (int): the dimensions of feature.
        momentum (float): momentum to update memory.
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """
    def __init__(self, num_clothes, feat_dim, momentum=0., scale=16, epsilon=0.1):
        super().__init__()
        self.num_clothes = num_clothes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale

        self.register_buffer('feature_memory', torch.zeros((num_clothes, feat_dim)))
        self.register_buffer('label_memory', torch.zeros(num_clothes, dtype=torch.int64) - 1)
        self.has_been_filled = False

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). 
        """
        # gather all samples from different GPUs to update memory.
        gathered_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gathered_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        self._update_memory(gathered_inputs.detach(), gathered_targets)

        inputs_norm = F.normalize(inputs, p=2, dim=1)
        memory_norm = F.normalize(self.feature_memory.detach(), p=2, dim=1)
        similarities = torch.matmul(inputs_norm, memory_norm.t()) * self.scale

        negtive_mask = 1 - positive_mask
        mask_identity = torch.zeros(positive_mask.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()

        if not self.has_been_filled:
            invalid_index = self.label_memory == -1
            positive_mask[:, invalid_index] = 0
            negtive_mask[:, invalid_index] = 0
            if sum(invalid_index.type(torch.int)) == 0:
                self.has_been_filled = True
                print('Memory bank is full')

        # compute log_prob
        exp_logits = torch.exp(similarities)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        mask = (1 - self.epsilon) * mask_identity + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (- mask * log_prob).sum(1).mean()
        
        return loss

    def _update_memory(self, features, labels):
        label_to_feat = {}
        for x, y in zip(features, labels):
            if y not in label_to_feat:
                label_to_feat[y] = [x.unsqueeze(0)]
            else:
                label_to_feat[y].append(x.unsqueeze(0))
        if not self.has_been_filled:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = feat
                self.label_memory[y] = y
        else:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = self.momentum * self.feature_memory[y] + (1. - self.momentum) * feat
                # self.embedding_memory[y] /= self.embedding_memory[y].norm()