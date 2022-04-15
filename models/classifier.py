import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter


__all__ = ['Classifier', 'NormalizedClassifier']


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y
        

class NormalizedClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,0,1e-5).mul_(1e5) 

    def forward(self, x):
        w = self.weight  

        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)

        return F.linear(x, w)



