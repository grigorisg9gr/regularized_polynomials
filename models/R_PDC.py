'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
from math import gcd
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.normalisation_scheme import IBN

from dropblock import DropBlock2D, LinearScheduler


def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif norm_local == 'a':
        norm_local = IBN
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local


def conv_separ(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, **kwargs):
    if isinstance(padding, int):
        padding = [padding] * 2
    n_groups, gcd1 = min(in_channels, out_channels), gcd(in_channels, out_channels)
    # # for 3x3 kernel, we define a separable convolution.
    if kernel_size == 3 and gcd1 == n_groups:
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 1), stride, (padding[0], 0), groups=n_groups, bias=False),
            nn.Conv2d(out_channels, out_channels, (1, 3), 1, (0, padding[1]), groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
        )
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias)
    return conv


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_activ=True, use_alpha=False, n_lconvs=0, 
                 norm_local=None, kern_loc=1, norm_layer=None, use_lactiv=False, norm_x=-1,
                 n_xconvs=0, what_lactiv=-1, kern_loc_so=3, use_uactiv=False, norm_bso=None,
                 use_only_first_conv=False, n_bsoconvs=0, init_a=0, planes_ho=None, 
                 use_conv_sep=False, div_factor=1, share_norm=False, **kwargs):
        """
        R-PDC residual block. 
        :param use_activ: bool; if True, use activation functions in the block.
        :param use_alpha: bool; if True, use a learnable parameter to regularize the contribution of the second order term.
        :param n_lconvs: int; the number of convolutional layers for the second order term.
        :param norm_local: int; the type of normalization scheme for the second order term.
        :param kern_loc: int;
        :param norm_layer: int; the type of normalization scheme for the first order term.
        :param use_lactiv: bool; if True, use activation functions for the second order term and 'x' (shortcut one).
        :param norm_x: int; the type of normalization scheme for the 'x' (shortcut one).
        :param n_xconvs: int; the number of convolutional layers for 'x' (shortcut one).
        :param what_lactiv: int; the type of the activation function for the second order term.
        :param kern_loc_so: int;
        :param use_uactiv: bool;
        :param norm_bso: int; the type of normalization scheme before the multiplication. 
        :param use_only_first_conv: bool;
        :param n_bsoconvs: int; the number of convolutional layers for the second order term before the multiplication.
        :param init_a:
        :param planes_ho: int; (Internal) planes in the higher-order terms.
        :param use_conv_sep: bool;
        :param div_factor:
        :param share_norm: bool; 
        :param kwargs:
        """                       
        super(BasicBlock, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self._norm_bso = get_norm(norm_bso)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs
        self.n_bsoconvs = n_bsoconvs
        self.n_lconvs = n_lconvs
        self.use_lactiv = self.use_activ and use_lactiv
        self.use_uactiv = self.use_activ and use_uactiv
        self.use_only_first_conv = use_only_first_conv
        self.use_conv_sep = use_conv_sep
        self.conv_layer = conv_separ if self.use_conv_sep else nn.Conv2d
        self.div_factor = div_factor
        self.share_norm = share_norm



        self.conv1 = self.conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        if norm_layer == 'a':
            self.bn1 = IBN(planes, ratio=0.8)
            self._norm_layer = nn.BatchNorm2d
        else:
            self.bn1 = self._norm_layer(planes)

        if not self.use_only_first_conv:
            self.conv2 = self.conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = self._norm_layer(planes)

        self.shortcut = nn.Sequential()
        planes1 = in_planes
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(self.expansion*planes)
            )
            planes1 = self.expansion * planes
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1) * init_a)
            self.monitor_alpha = []
        self.normx = self._norm_x(planes1)
        if 1:
            if what_lactiv == -1:
                ac1 = nn.ReLU(inplace=True)
            elif what_lactiv == 1:
                ac1 = nn.Softmax2d()
            elif what_lactiv == 2:
                ac1 = nn.LeakyReLU(inplace=True)
            elif what_lactiv == 3:
                ac1 = torch.tanh
        self.lactiv = partial(ac1) if self.use_lactiv else lambda x: x
        # # check the output planes for higher-order terms.
        if planes_ho is None or planes_ho < 0:
            planes_ho = planes1
        print('(Internal) planes in the higher-order terms: {}.'.format(planes_ho))
        # # define 'local' convs, i.e. applied only to second order term after the multiplication.
        self.def_local_convs(planes_ho, n_lconvs, kern_loc, self._norm_local, key='l', out_planes=planes)
        self.def_local_convs(planes1, n_bsoconvs, kern_loc, self._norm_bso, key='bso')
        #self.def_local_convs(planes1, n_xconvs, kern_loc, self._norm_x, key='x')
        self.def_convs_so(planes1, kern_loc_so, self._norm_x, key=1, out_planes=planes_ho)
        self.def_convs_so(planes1, kern_loc_so, self._norm_x, key=2, out_planes=planes_ho, share_norm=self.share_norm)
        self.uactiv = partial(ac1) if self.use_uactiv else lambda x: x

        print('norm layer: {}'.format(norm_layer))
        print('norm_x: {}'.format(norm_x))
        print('norm_local: {}'.format(norm_local))

    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        if not self.use_only_first_conv:
            out = self.bn2(self.conv2(out))
        out1 = out + self.shortcut(x)
        # # normalize the x (shortcut one).
        x1 = self.normx(self.shortcut(x))
        x1 = self.apply_local_convs(x1, self.n_bsoconvs, key='bso')
        x2 = self.apply_convs_so(x1, key=1)
        x3 = self.apply_convs_so(x1, key=2)
        out_so = x2 * x3
        if self.div_factor != 1:
            out_so = out_so * 1. / self.div_factor
        out_so = self.apply_local_convs(out_so, self.n_lconvs, key='l')
        if self.use_alpha:
            out1 += self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out_so
        out = self.activ(out1)
        return out

    def def_local_convs(self, planes, n_lconvs, kern_loc, func_norm, key='l', typet='conv', out_planes=None):
        """ Aux function to define the local conv/fc layers. """
        if out_planes is None:
            out_planes = planes
        if n_lconvs > 0:
            s = '' if n_lconvs == 1 else 's'
            print('Define {} local {}{}{}.'.format(n_lconvs, key, typet, s))
            if typet == 'conv':
                convl = partial(self.conv_layer, in_channels=planes, out_channels=out_planes,
                                kernel_size=kern_loc, stride=1, padding=int(kern_loc > 1), bias=False)
            else:
                convl = partial(nn.Linear, planes, planes)
            for i in range(n_lconvs):
                setattr(self, '{}{}{}'.format(key, typet, i), convl())
                setattr(self, '{}bn{}'.format(key, i), func_norm(out_planes))

    def apply_local_convs(self, out_so, n_lconvs, key='l'):
        if n_lconvs > 0:
            for i in range(n_lconvs):
                out_so = getattr(self, '{}conv{}'.format(key, i))(out_so)
                out_so = self.lactiv(getattr(self, '{}bn{}'.format(key, i))(out_so))
        return out_so
    
    def def_convs_so(self, planes, kern_loc, func_norm, key=1, out_planes=None, share_norm=False):
        """ Aux function to define the conv layers for the second order. """
        if out_planes is None:
            out_planes = planes
        convl = partial(self.conv_layer, in_channels=planes, out_channels=out_planes,
                        kernel_size=kern_loc, stride=1, padding=int(kern_loc > 1), bias=False)
        setattr(self, 'u_conv{}'.format(key), convl())
        if key > 1 and share_norm:
            pass
        else:
            setattr(self, 'ubn{}'.format(key), func_norm(out_planes))

    def apply_convs_so(self, input1, key=1):
        # # second order convs.
        out_uo = getattr(self, 'u_conv{}'.format(key))(input1)
        key1 = key if key > 1 and not self.share_norm else 1
        out_uo = self.uactiv(getattr(self, 'ubn{}'.format(key1))(out_uo))
        return out_uo


class R_PDC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None, out_activ=False, pool_adapt=False, 
                 n_channels=[64, 128, 256, 512], planes_ho=None, ch_in=3, **kwargs):
        super(R_PDC, self).__init__()

        # define max pooling and dropblock
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.dropblock1 = DropBlock2D(block_size=7, drop_prob=0.1)
        self.dropblock2 = DropBlock2D(block_size=7, drop_prob=0.1)

        self.in_planes = n_channels[0]
        if norm_layer is None:
            n_norm_layer = nn.BatchNorm2d
        elif isinstance(norm_layer, list):
            n_norm_layer = nn.BatchNorm2d
        else:
            norm_layer = get_norm(norm_layer)

        self._norm_layer = n_norm_layer
        self.out_activ = out_activ
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)
        if planes_ho is None or isinstance(planes_ho, int):
            planes_ho = [planes_ho] * len(n_channels)

        self.conv1 = nn.Conv2d(ch_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, planes_ho=planes_ho[0], norm_layer=norm_layer[0], **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, planes_ho=planes_ho[1], norm_layer=norm_layer[1], **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, planes_ho=planes_ho[2], norm_layer=norm_layer[2], **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, planes_ho=planes_ho[3], norm_layer=norm_layer[3], **kwargs)
        if len(n_channels) > 4:
            print('Using additional blocks (org. 4): ', len(n_channels))
            for i in range(len(n_channels) - 4):
                j = i + 4
                setattr(self, 'layer{}'.format(j + 1), self._make_layer(block, n_channels[j], num_blocks[j], stride=1, **kwargs))
        self.linear = nn.Linear(n_channels[-1] * block.expansion, num_classes)
        # # if linear case and requested, include an output non-linearity.
        cond = self.out_activ and self.activ(-100) == -100
        self.oactiv = partial(nn.ReLU(inplace=True)) if cond else lambda x: x
        print('output non-linearity: #', self.out_activ, cond)


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
        if len(self.n_channels) > 4:
            for i in range(len(self.n_channels) - 4):
                out = getattr(self, 'layer{}'.format(i + 5))(out)
        if self.out_activ:
            out = self.oactiv(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

def R_PDC_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return R_PDC(BasicBlock, num_blocks, **kwargs)


def test():
    net = R_PDC_wrapper()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
