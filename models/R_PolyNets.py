'''
R-PolyNets in PyTorch.

Reference:
[1] Grigorios Chrysos, Bohan Wang, Jiankang Deng, Volkan Cevher. "Regularization of polynomial networks for image recognition", CVPR'23. 

'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from src.iterative_normalization import IterNorm
from src.normalisation_scheme import SwitchNorm2d, DBN, IBN, BatchInstanceNorm2d 
from dropblock import DropBlock2D, LinearScheduler


def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif norm_local == 2:
       norm_local = IterNorm
    # # # New normalization schemes
    elif norm_local == 3:
        norm_local = SwitchNorm2d
    elif norm_local == 4:
        norm_local = DBN
    elif norm_local == 5:
        norm_local = BatchInstanceNorm2d
    elif norm_local ==6:
        al = nn.Parameter(torch.tensor([[1 / math.sqrt(6)]])).to('cuda') 
        norm_local = lambda a: lambda x: al * x 
    elif norm_local == 'a':
        norm_local = IBN
        
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_activ=False, use_alpha=True, n_lconvs=1,
                 norm_local=None, kern_loc=1, norm_layer=2, norm_x=-1, **kwargs):
        """
        R-PolyNets residual block. 
        :param use_activ: bool; if True, use activation functions in the block.
        :param use_alpha: bool; if True, use a learnable parameter to regularize the contribution of the second order term.
        :param use_beta:  bool; if True, use a learnable parameter to regularize the contribution of the first order term.
        :param n_lconvs: int; the number of convolutional layers for the second order term.
        :param norm_local: int; the type of normalization scheme for the second order term.
        :param kern_loc: int
        :param norm_layer: int; the type of normalization scheme for the first order term.
        :param norm_x: int; the type of normalization scheme for the 'x' (shortcut one).
        :param kwargs:
        """                 
        super(BasicBlock, self).__init__()
        
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # select the normalization scheme for the block
        if norm_layer == 2:
            self.bn1 = IterNorm(planes)
            self._norm_layer = nn.BatchNorm2d
        elif norm_layer == 'a':
            self.bn1 = IBN(planes, ratio=0.8)
            self._norm_layer = nn.BatchNorm2d
        else:
            self.bn1 = self._norm_layer(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self._norm_layer(planes)

        self.shortcut = nn.Sequential()
        planes1 = in_planes
        if stride != 1 or in_planes != self.expansion*planes:
            if norm_layer == 6:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    self._norm_layer(num_channels = self.expansion*planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    self._norm_layer(self.expansion*planes)
                )
            planes1 = self.expansion * planes
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.use_alpha = use_alpha
        
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.monitor_alpha = []

        self.normx = self._norm_x(planes1)
        # # define 'local' convs, i.e. applied only to second order term, either
        # # on x or after the multiplication.
        self.def_local_convs(planes, n_lconvs, kern_loc, self._norm_local, key='l')
            
        print('norm layer: {}'.format(norm_layer))
        print('norm_x: {}'.format(norm_x))
        print('norm_local: {}'.format(norm_local))
        
    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # multiple out with beta
        out1 = out + self.shortcut(x)
        # # normalize the x (shortcut one).
        x1 = self.normx(self.shortcut(x))
        # second order
        out_so = out * x1

        out_so = self.apply_local_convs(out_so, self.n_lconvs, key='l')        
        if self.use_alpha:
            out1 += self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out_so
       
        out = self.activ(out1)
        return out

    def def_local_convs(self, planes, n_lconvs, kern_loc, func_norm, key='l', typet='conv'):
        """ Aux function to define the local conv/fc layers. """
        if n_lconvs > 0:
            s = '' if n_lconvs == 1 else 's'
            print('Define {} local {}{}{}.'.format(n_lconvs, key, typet, s))
            if typet == 'conv':
                convl = partial(nn.Conv2d, in_channels=planes, out_channels=planes,
                                kernel_size=kern_loc, stride=1, padding=int(kern_loc > 1), bias=False)
            else:
                convl = partial(nn.Linear, planes, planes)
            for i in range(n_lconvs):
                setattr(self, '{}{}{}'.format(key, typet, i), convl())
                setattr(self, '{}bn{}'.format(key, i), func_norm(planes))

    def apply_local_convs(self, out_so, n_lconvs, key='l'):
        if n_lconvs > 0:
            for i in range(n_lconvs):
                out_so = getattr(self, '{}conv{}'.format(key, i))(out_so)
                out_so = getattr(self, '{}bn{}'.format(key, i))(out_so)

        return out_so


class R_PolyNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None,
                 pool_adapt=False, n_channels=[64, 128, 256, 512], ch_in=3, **kwargs):
        super(R_PolyNets, self).__init__()

        self.snapshots_layer1 = []
        self.snapshots_layer2 = []
        self.snapshots_layer3 = []
        self.snapshots_layer4 = []
        self.tsne_layer3 = []
        self.tsne_layer4 = []
        self.temp_debugging = 0
        self.label = np.empty(128)
        self.epoch = 0
        
        # # define max pooling and dropblock
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.dropblock1 = DropBlock2D(block_size=7, drop_prob=0.1)
        self.dropblock2 = DropBlock2D(block_size=7, drop_prob=0.1)

        self.in_planes = n_channels[0]
            
        if isinstance(norm_layer, list):
            n_norm_layer = nn.BatchNorm2d
            
        self._norm_layer = n_norm_layer
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)

        self.conv1 = nn.Conv2d(ch_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, norm_layer=norm_layer[0], **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, norm_layer=norm_layer[1], **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, norm_layer=norm_layer[2], **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, norm_layer=norm_layer[3], **kwargs)
        self.linear = nn.Linear(n_channels[-1] * block.expansion, num_classes)
        
        # # define the hook
        for name, layer in self.named_children():
            layer.__name__ = name
        
        # 16/N zero-mean gaussian initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                print('new initialization!')
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(0.25 / (m.conv1.weight.shape[1] * np.prod(m.conv1.weight.shape[2:]))) * 8)
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(0.25 / (m.conv2.weight.shape[1] * np.prod(m.conv2.weight.shape[2:]))) * 8)
                nn.init.normal_(m.lconv0.weight, mean=0, std=np.sqrt(0.25 / (m.lconv0.weight.shape[1] * np.prod(m.lconv0.weight.shape[2:]))) * 8)

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
                
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                               norm_layer=norm_layer, **kwargs))
            self.in_planes = planes * block.expansion
        # # cheeky way to get the activation from the layer1, e.g. in no activation case.
        self.activ = layers[0].activ
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.dropblock1(self.maxpool(self.layer1(out)))
        out = self.dropblock2(self.maxpool(self.layer2(out)))
        out = self.maxpool(self.layer3(out))
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def R_PolyNets_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return R_PolyNets(BasicBlock, num_blocks, **kwargs)


def test():
    net = R_PolyNets_wrapper()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

