# inflate 2D modules to 3D modules
import torch
import torch.nn as nn
from torch.nn import functional as F


def inflate_conv(conv2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """
    Args:
        time_dim: final time dimension of the features
    """
    linear3d = nn.Linear(linear2d.in_features * time_dim,
                               linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = nn.Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    # In pytorch 0.2.0 the 2d and 3d versions of batch norm
    # work identically except for the check that verifies the
    # input dimensions

    batch3d = nn.BatchNorm3d(batch2d.num_features)
    # retrieve 3d _check_input_dim function
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d,
                 time_dim=1,
                 time_padding=0,
                 time_stride=None,
                 time_dilation=1):
    kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
    padding = (time_padding, pool2d.padding, pool2d.padding)
    if time_stride is None:
        time_stride = time_dim
    stride = (time_stride, pool2d.stride, pool2d.stride)
    if isinstance(pool2d, nn.MaxPool2d):
        dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
        pool3d = nn.MaxPool3d(
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
            ceil_mode=pool2d.ceil_mode)
    elif isinstance(pool2d, nn.AvgPool2d):
        pool3d = nn.AvgPool3d(kernel_dim, stride=stride)
    else:
        raise ValueError(
            '{} is not among known pooling classes'.format(type(pool2d)))
    return pool3d


class MaxPool2dFor3dInput(nn.Module):
    """
    Since nn.MaxPool3d is nondeterministic operation, using fixed random seeds can't get consistent results.
    So we attempt to use max_pool2d to implement MaxPool3d with kernelsize (1, kernel_size, kernel_size).
    """
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous() # b, t, c, h, w
        x = x.view(b*t, c, h, w)
        # max pooling
        x = self.maxpool(x)
        _, _, h, w = x.size()
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        return x