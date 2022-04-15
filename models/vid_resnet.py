import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from models.utils import inflate
from models.utils import c3d_blocks
from models.utils import nonlocal_blocks


__all__ = ['AP3DResNet50', 'AP3DNLResNet50', 'NLResNet50', 'C2DResNet50', 
           'I3DResNet50', 
          ] 


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super().__init__()
        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet503D(nn.Module):
    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT

        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride=(1, 1)
            resnet2d.layer4[0].downsample[0].stride=(1, 1) 

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        # self.maxpool = inflate.MaxPool2dFor3dInput(kernel_size=resnet2d.maxpool.kernel_size,
        #                                            stride=resnet2d.maxpool.stride,
        #                                            padding=resnet2d.maxpool.padding,
        #                                            dilation=resnet2d.maxpool.dilation)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], \
                                             nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], \
                                             nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], \
                                             nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], \
                                             nonlocal_idx=nl_idx[3], nonlocal_channels=2048)

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, \
                                       temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        # spatial max pooling
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        # temporal avg pooling
        x = x.mean(1)
        f = self.bn(x)

        return f


def C2DResNet50(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def AP3DResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def I3DResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[],[],[]]

    return ResNet503D(config, c3d_blocks.I3D, c3d_idx, nl_idx, **kwargs)


def AP3DNLResNet50(config, **kwargs):
    c3d_idx = [[],[0, 2],[0, 2, 4],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)


def NLResNet50(config, **kwargs):
    c3d_idx = [[],[],[],[]]
    nl_idx = [[],[1, 3],[1, 3, 5],[]]

    return ResNet503D(config, c3d_blocks.APP3DC, c3d_idx, nl_idx, **kwargs)
