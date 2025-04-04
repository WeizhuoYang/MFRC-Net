"""
Weizhuoyang 2024.9
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
