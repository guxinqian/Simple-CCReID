import torch
from torch import nn


class CrossEntropyWithLabelSmooth(nn.Module):
    """ Cross entropy loss with label smoothing regularization.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. In CVPR, 2016.
    Equation: 
        y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): a hyper-parameter in the above equation.
    """
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        _, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss
