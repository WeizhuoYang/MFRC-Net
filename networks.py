#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""
import torch
import torch.nn as nn
import sys
from torchsummary import summary
import torch.nn.init as init
import torch.nn.functional as F
from typing import Tuple, Optional
current_module = sys.modules[__name__]
debug = False
#%% Support classes for MFRCNet Implementation
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class VarLayer(nn.Module):
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

def _get_padding(kernel_size, stride=1, dilation=1, **_):
    if isinstance(kernel_size, tuple):
        kernel_size = max(kernel_size)
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2dSameExport(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = (0, 0)

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:], self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size

        if self.pad is not None:
            x = self.pad(x)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

def get_padding_value(padding, kernel_size, **kwargs):
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if _is_static_pad(kernel_size, **kwargs):
                padding = _get_padding(kernel_size,**kwargs)
            else:
                padding = 0
                dynamic = True
            padding = 0
        else:
            padding = _get_padding(kernel_size,**kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        if isinstance(kernel_size,tuple):
            padding = (0,padding)
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)

class MixedConv2d(nn.ModuleDict):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
                 )
        self.splits = in_splits
    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x

#%% MFRCNet
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

class TMSRC(nn.Module):
    def __init__(self,in_chan,num_Feat):
        super(TMSRC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 63), groups=in_chan, padding=(0, 31)),
            nn.BatchNorm2d(num_Feat),
        )
        self.mixConv2d = nn.Sequential(
            MixedConv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=[(1,15),(1,25),(1,31),(1,125)],
                         stride=1, padding='', dilation=1, depthwise=False),
            ChannelShuffle(groups=num_Feat // 9),
            nn.BatchNorm2d(num_Feat),
        )
    def forward(self,x):
        y = x
        x = self.conv1(x)
        x += self.mixConv2d(y)
        return x

class CrossDC(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def __init__(self,in_chan,num_Feat,dilatability,nChan):
        super(CrossDC, self).__init__()
        self.qm = nn.Sequential(
            TMSRC(in_chan=9, num_Feat=num_Feat),
            self.SCB(in_chan=num_Feat, out_chan=num_Feat * dilatability, nChan=int(nChan)),
        )
        self.shortcut=nn.Sequential()
        if in_chan!=num_Feat*dilatability:
            self.shortcut=nn.Sequential(
                 nn.Conv2d(in_channels=in_chan,out_channels=num_Feat,kernel_size=(1,1)),
                 self.SCB(in_chan=num_Feat,out_chan=num_Feat*dilatability,nChan=int(nChan)),
            )
    def forward(self,x):
        residual = x
        x = self.qm(x)
        x += self.shortcut(residual)
        return x

class MFRCNet(nn.Module):
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,nBands=9,temporalLayer='VarLayer', num_Feat=18, dilatability=8, dropoutP=0.5, *args, **kwargs):
        super(MFRCNet, self).__init__()
        self.strideFactor = 4
        self.nBands = nBands
        self.mix=CrossDC(in_chan=9,num_Feat=num_Feat,dilatability=dilatability,nChan=nChan)
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)
        size = self.get_size(nChan, nTime)
        self.fc = self.LastBlock(size[1],nClass)

        init.xavier_uniform_(self.fc[0].weight)

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.mix(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        #print(x.shape)
        return c, f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        x = self.mix(data)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()


if __name__ == "__main__":
    net = MFRCNet(nChan=22, nTime=1000).cuda()
    print(net)
    summary(net, (9, 22, 1000))
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params}")

    # Params  FLOPs
    from thop import profile
    input_data = torch.randn(16, 9, 22, 1000).cuda()
    flops, params = profile(net, inputs=(input_data,))
    print(f"FLOPs: {flops / 1e6} M")
    print(f"Params: {params / 1e6} M")
'''


#%% multiscale

class MSconv_t(nn.Module):
    def __init__(self, in_chan, num_Feat):
        super(MSconv_t, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 15), groups=in_chan, padding=(0, 7)),
            nn.BatchNorm2d(num_Feat),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 31), groups=in_chan, padding=(0, 15)),
            nn.BatchNorm2d(num_Feat),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 63), groups=in_chan, padding=(0, 31)),
            nn.BatchNorm2d(num_Feat),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 125), groups=in_chan, padding=(0, 62)),
            nn.BatchNorm2d(num_Feat),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=num_Feat, kernel_size=(1, 25), groups=in_chan, padding=(0, 12)),
            nn.BatchNorm2d(num_Feat),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        out = torch.cat([x1, x2, x3, x4,x5], dim=1)
        print(out.shape)
        return out

class Resscb(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def __init__(self,in_chan,num_Feat,dilatability,nChan):
        super(Resscb, self).__init__()
        self.qm = nn.Sequential(
            MSconv_t(in_chan=9, num_Feat=num_Feat),
            self.SCB(in_chan=num_Feat*5, out_chan=num_Feat * dilatability, nChan=int(nChan)),
        )
        self.shortcut=nn.Sequential()
        if in_chan!=num_Feat*dilatability:
            self.shortcut=nn.Sequential(
                 nn.Conv2d(in_channels=in_chan,out_channels=num_Feat,kernel_size=(1,1)),
                 self.SCB(in_chan=num_Feat,out_chan=num_Feat*dilatability,nChan=int(nChan)),
            )
    def forward(self,x):
        residual = x
        x = self.qm(x)
        x += self.shortcut(residual)
        return x


class MSNet(nn.Module):
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,nBands=9,temporalLayer='VarLayer', num_Feat=36, dilatability=10, dropoutP=0.5, *args, **kwargs):
        super(MSNet, self).__init__()
        self.strideFactor = 4
        self.nBands = nBands
        self.mix=Resscb(in_chan=9,num_Feat=num_Feat,dilatability=dilatability,nChan=nChan)
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)
        size = self.get_size(nChan, nTime)
        self.fc = self.LastBlock(size[1],nClass)
        init.xavier_uniform_(self.fc[0].weight)  # 全连接层之前调用这个函数初始化

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.mix(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        return c, f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        x = self.mix(data)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()

if __name__ == "__main__":
    net = MSNet(nChan=22, nTime=1000).cuda()
    print(net)
    summary(net, (9, 22, 1000))
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params}")

    # Params  FLOPs
    from thop import profile
    input_data = torch.randn(16, 9, 22, 1000).cuda()
    flops, params = profile(net, inputs=(input_data,))
    print(f"FLOPs: {flops / 1e6} M")
    print(f"Params: {params / 1e6} M")
'''
#%% FBMSNet_Inception
class MFRC_inception(nn.Module):
    def SCB(self, in_chan, out_chan, nChan, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
            Conv2dWithConstraint(in_chan, out_chan, (nChan, 1), groups=in_chan,
                                 max_norm=2, doWeightNorm=doWeightNorm, padding=0),
            nn.BatchNorm2d(out_chan),
            swish()
        )
    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
            LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs),
            nn.LogSoftmax(dim=1))
    def __init__(self, nChan, nTime, nClass=4,sampling_rate=250, temporalLayer='LogVarLayer', num_Feat=36, dilatability=8, dropoutP=0.5, *args, **kwargs):
        # input_size: channel x datapoint
        super(MFRC_inception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625]
        self.strideFactor = 4

        self.Tception1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[0] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.Tception2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[1] * sampling_rate-1)), stride=1, padding=(0,int(self.inception_window[1] * sampling_rate/2-1))),
            nn.BatchNorm2d(9),
        )
        self.Tception3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[2] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.Tception4 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=(1, int(self.inception_window[3] * sampling_rate)), stride=1, padding=(0,int(self.inception_window[3] * sampling_rate/2))),
            nn.BatchNorm2d(9),
        )
        self.scb = self.SCB(in_chan=36, out_chan=288, nChan=int(nChan))

        self.shortcut = nn.Sequential()
        if 9!=num_Feat*dilatability:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=9,out_channels=num_Feat,kernel_size=(1,1)),
                 self.SCB(in_chan=num_Feat,out_chan=num_Feat*dilatability,nChan=int(nChan)),
            )
        self.temporalLayer = current_module.__dict__[temporalLayer](dim=3)

        size = self.get_size(nChan, nTime)

        self.fc = self.LastBlock(size[1],nClass)

    def forward(self, x):
        if len(x.shape) == 5:
            x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=1)
        y = self.Tception4(x)
        out = torch.cat((out, y), dim=1)
        y = self.scb(out)
        out = y + self.shortcut(x)
        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[3] / self.strideFactor)])
        out = self.temporalLayer(out)
        f = torch.flatten(out, start_dim=1)
        c = self.fc(f)
        return c,f

    def get_size(self, nChan, nTime):
        data = torch.ones((1, 9, nChan, nTime))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=1)
        y = self.Tception4(data)
        out = torch.cat((out, y), dim=1)
        z= self.scb(out)
        x = z + self.shortcut(data)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim=1)
        return x.size()


from torchsummary import summary

#%%
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat

class Conv_block(nn.Module):
    def __init__(self, F1=16, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
        super(Conv_block, self).__init__()
        F2 = F1 * D
        self.block1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.block2 = Conv2dWithConstraint(F1, F1 * D, (in_chans, 1), bias=False, groups=F1,
                                           doWeightNorm=True, max_norm=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu = nn.ELU()
        self.avg1 = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropout)
        self.block3 = nn.Conv2d(F2, F2, (1, 16), bias=False, padding='same')
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg2 = nn.AvgPool2d((1, poolSize))
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.bn1(self.block1(x))
        x = self.elu(self.bn2(self.block2(x)))
        x = self.avg1(x)
        x = self.dropout(x)
        x = self.elu(self.bn3(self.block3(x)))
        x = self.avg2(x)
        x = self.dropout(x)

        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn1, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, padding='same') if n_inputs != n_outputs else None
        self.relu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class mha_block(nn.Module):
    def __init__(self, key_dim=8, num_heads=2, dropout=0.5):
        super(mha_block, self).__init__()

        self.LayerNorm = nn.LayerNorm(32, eps=1e-6)
        self.bn = nn.BatchNorm1d(32)
        self.mha = MultiHeadAttention_atc(32, num_heads, dropout, key_dim=8)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        res = x
        # x = self.LayerNorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        x = self.mha(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = res + x

        return x


class MultiHeadAttention_atc(nn.Module):
    def __init__(self, emb_size, num_heads, dropout, key_dim):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, key_dim * num_heads)
        self.queries = nn.Linear(emb_size, key_dim * num_heads)
        self.values = nn.Linear(emb_size, key_dim * num_heads)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(key_dim * num_heads, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TCN_block(nn.Module):
    def __init__(self, input_dimension, depth, kernel_size, filters, dropout):
        super(TCN_block, self).__init__()

        self.block = TemporalBlock(n_inputs=input_dimension, n_outputs=filters, stride=1, dilation=1,
                              kernel_size=kernel_size, padding=(kernel_size-1) * 1, dropout=dropout)

        layers = []
        for i in range(depth - 1):
            dilation_size = 2 ** (i + 1)
            layers += [TemporalBlock(input_dimension, filters, stride=1, dilation=dilation_size,
                                     kernel_size=kernel_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.network(x)
        return x


class ATCNet(nn.Module):
    def __init__(self, nChan=22, nClass=4, nTime=1000, in_samples=1125, n_windows=5, attention='mha',
                 eegn_F1=16, eegn_D=2, eegn_kernelSize=64, eegn_poolSize=7, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                 fuse='average', *args, **kwargs):
        super(ATCNet, self).__init__()

        n_classes = nClass
        in_chans = nChan
        regRate = .25
        numFilters = eegn_F1
        F2 = numFilters * eegn_D
        self.n_windows = n_windows
        self.fuse = fuse

        self.block1 = Conv_block(F1=eegn_F1, D=eegn_D, kernLength=eegn_kernelSize, poolSize=eegn_poolSize,
                                 in_chans=in_chans, dropout=eegn_dropout)

        self.dense_list = nn.ModuleList([LinearWithConstraint(tcn_filters, n_classes,
                                                doWeightNorm=True, max_norm=regRate) for i in range(self.n_windows)])

        self.attention_block_list = nn.ModuleList([mha_block() for i in range(self.n_windows)])

        self.tcn_block_list = nn.ModuleList([TCN_block(input_dimension=F2, depth=tcn_depth, kernel_size=tcn_kernelSize,
                                   filters=tcn_filters, dropout=tcn_dropout) for i in range(self.n_windows)])

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # if len(x.shape) == 5:
        #     x = torch.squeeze(x, dim=4)
        x1 = self.block1(x).squeeze(2)
        sw_concat = []
        for i in range(self.n_windows):
            st = i
            end = x1.shape[2] - self.n_windows + i + 1
            x2 = x1[:, :, st:end]

            # attention or identity
            x2 = self.attention_block_list[i](x2)
            # TCN
            x3 = self.tcn_block_list[i](x2)
            # Get feature maps of the last sequence
            x3 = x3[:, :, -1]
            # Outputs of sliding window: Average_after_dense or concatenate_then_dense
            sw_concat.append(self.dense_list[i](x3))

        if len(sw_concat) > 1:
            sw_concat = torch.mean(torch.stack(sw_concat, dim=1), dim=1)
        else:
            sw_concat = sw_concat[0]



        return self.logsoftmax(sw_concat),sw_concat        # 可视化！！！




