'''
D-PolyNets in PyTorch.

Reference:
[1] Grigorios Chrysos, Bohan Wang, Jiankang Deng, Volkan Cevher. "Regularization of polynomial networks for image recognition", CVPR'23. 

'''
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iterative_normalization import IterNorm
from src.normalisation_scheme import IBN
from dropblock import DropBlock2D, LinearScheduler


def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif norm_local == 2:
       norm_local = IterNorm        
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_activ=False, use_alpha=False, n_lconvs=0, 
                 norm_local=None, kern_loc=1, norm_layer=None, norm_x=-1, n_xconvs=0, append_bef_norm=True,
                 use_pr_conv=False, kern_prev=1, norm_pr=-1, dropout=0, train=True, limit_pr=None,
                 sum_prev=False, additive_gn=0, skip_con_dense=False,
                 use_sep_short=False, block_id=None, use_sep_short_all=False, use_xlconv=True, 
                 use_full_prev_poly=False, use_theta=False, random_scaling=False, start_block=0, **kwargs):
        """
        D-PolyNets residual block. The difference from Pi-net residual block, is that in each block we
        also perform one dense connection from previous blocks. That means that during forward, we pass two inputs.
        :param use_activ: bool; if True, use activation functions in the block.
        :param use_alpha: bool; if True, use a learnable parameter to regularize each previous polynomial in the current Hadamard product.
        :param n_lconvs: int; the number of convolutional layers for the higher order term.
        :param norm_local: int; the type of normalization scheme for the higher order term.
        :param kern_loc:
        :param norm_layer: int; the type of normalization scheme for the first order term.
        :param norm_x: int; the type of normalization scheme for the 'x' (shortcut one).
        :param n_xconvs:
        :param use_pr_conv: bool; if True, use convolutions in the dense representations.
        :param kern_prev:
        :param norm_pr: int; the type of normalization scheme for the dense representations.
        :param dropout: float; if positive, it uses that as a probability threshold to skip a dense connection
            multiplication.
        :param limit_pr: int; the number of previous Hadamard connections to include; if None, it
            multiplies with all of them.
        :param sum_prev: bool; if True use a skip in the dense product.
        :param additive_gn: If 0, there is no Gaussian noise in the dense connections; if > 0, 
            it samples an Gaussian noise in every iteration for every connection.
        :param kwargs:
        """
        super(BasicBlock, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self._norm_prev = get_norm(norm_pr)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs
        self.n_xconvs = n_xconvs
        self.append_bef_norm = append_bef_norm
        self.use_pr_conv = use_pr_conv
        self.is_training = train
        self.dropout = dropout
        if limit_pr is None:
            limit_pr = 1000
        self.limit_prev = limit_pr
        assert self.limit_prev > 0
        self.sum_prev = sum_prev
        self.additive_gn = additive_gn
        self.skip_con_dense = skip_con_dense

        # # self.use_fd: index of the start block to transmit the polynomial. The default is zero.
        self.use_fd = int(start_block)
        self.use_sep_short = use_sep_short
        self.block_id = block_id
        self.use_sep_short_all = use_sep_short_all
        self.use_xlconv = use_xlconv
        self.use_full_prev_poly = use_full_prev_poly

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
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
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(self.expansion*planes)
            )
            planes1 = self.expansion * planes

        if stride != 1 or in_planes != self.expansion*planes and self.use_sep_short:
            shortcut = lambda: nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    self._norm_layer(self.expansion*planes)
                )
            self.cond_short = self.use_sep_short and self.block_id > 0 and self.use_sep_short_all
            if self.cond_short:
                self.maxl = self.block_id if self.limit_prev is None else min(self.block_id, self.limit_prev)
                for id1 in range(self.maxl):
                    setattr(self, 'shortcut_sep{}'.format(id1), shortcut())
                self.shortcut_sep = self.shortcut    
            else:
                self.shortcut_sep = shortcut()
        elif not self.use_sep_short:
            self.shortcut_sep = self.shortcut
        else:
            self.shortcut_sep = nn.Sequential()
        self.activ = partial(nn.ReLU(inplace=True)) if self.use_activ else lambda x: x
        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1))
            self.monitor_alpha = []
        self.normx = self._norm_x(planes1)
        self.normpr = self._norm_prev(planes1)
        # # define 'local' convs, i.e. applied only to second order term, either
        # # on x or after the multiplication.
        self.def_local_convs(planes, n_lconvs, kern_loc, self._norm_local, key='l')
        if self.use_xlconv:
            self.def_local_convs(planes1, n_xconvs, kern_loc, self._norm_x, key='x')
        if self.use_pr_conv:
            self.pr_conv = nn.Conv2d(planes1, planes1, kernel_size=kern_prev, stride=1, padding=int(kern_prev > 1))

        self.use_theta = use_theta
        self.random_scaling = random_scaling
        print('block id: {}'.format(self.block_id))
        print('norm layer: {}'.format(norm_layer))
        print('norm_x: {}'.format(norm_x))
        print('norm_local: {}'.format(norm_local))
        print('norm_prev: {}'.format(norm_pr))
        print('use active: {}'.format(self.use_activ))
        print('drop out: {}'.format(self.dropout))
        print('use xconv: {}'.format(self.use_xlconv))
        print('use pr conv: {}'.format(self.use_pr_conv))
        print('use sep short: {}'.format(self.use_sep_short))
        print('use sep short all: {}'.format(self.use_sep_short_all))
        print('sum prev: {}'.format(self.sum_prev))
        print('add gaussian: {}'.format(self.additive_gn))
        print('skip dense connection: {}'.format(self.skip_con_dense))
        print('use full polynomials: {}'.format(self.use_full_prev_poly))
        print('use theta: {}'.format(self.use_theta))
        print('use alpha: {}'.format(self.use_alpha))
        print('use random scaling: {}'.format(self.random_scaling))
        print('use limit: {}'.format(self.limit_prev))
        print('start block: {}'.format(self.use_fd))


        if self.dropout>0:
          self.beta = 1/(1-self.dropout)
        else:
          self.beta = 1.0
        print('beta: {}'.format(self.beta))         
        
        if self.use_theta and self.skip_con_dense:
          self.theta = nn.Parameter(torch.zeros(1))
          print(self.theta)
        else:
          self.theta = 1.0



    def forward(self, xa):
        x, second_ords = xa[0], xa[1]
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out1 = out + self.shortcut(x)
        # # normalize the x (shortcut one).
        x1 = self.normx(self.shortcut(x))
        if self.use_xlconv:
            x1 = self.apply_local_convs(x1, self.n_xconvs, key='x')
        if len(second_ords) > 0:
            # # check if there is any down-sampling either spatial or channel-wise.
            cond = not (second_ords[0].shape[1] == out.shape[1] and second_ords[0].shape[2] == out.shape[2])
            if cond:
                # # perform downsampling and augment the list.
                new_list = []
                # # we reverse the second_ords below for the case of cond_short==True, i.e. the case that
                # # we need to apply a different shortcut connection for each; since below the 'last'
                # # second_ords are used first, we need those with the separate shortcuts. This is reversed
                # # few lines below when assigning the new_list. 
                for cnt, prev_so in enumerate(second_ords[::-1]):
                    if self.cond_short and cnt < self.maxl:
                        prev_so = getattr(self, 'shortcut_sep{}'.format(cnt))(prev_so)
                    else:
                        prev_so = self.shortcut_sep(prev_so)
                    new_list.append(prev_so)
                second_ords = new_list[::-1]
        out0 = 1

        if len(second_ords) <= self.use_fd:
            out0 = 1
        elif len(second_ords) < self.limit_prev:
            for cnt, prev_so in enumerate(second_ords[::-1][self.use_fd:]):
                if self.additive_gn > 0 and self.training:
                    prev_so = prev_so + self.additive_gn * torch.randn(*prev_so.shape).to(x.device)
                if cnt == 0:
                    out0 = prev_so
                elif self.sum_prev:
                    out0 = out0 + out0 * prev_so                   
                else:
                    out0 = out0 * prev_so
        else:
            for cnt, prev_so in enumerate(second_ords[::-1][self.use_fd: self.limit_prev]):
                if self.additive_gn > 0 and self.training:
                    prev_so = prev_so + self.additive_gn * torch.randn(*prev_so.shape).to(x.device)
                if cnt == 0:
                    out0 = prev_so
                elif self.sum_prev:
                    out0 = out0 + out0 * prev_so                   
                else:
                    out0 = out0 * prev_so
    
        if not isinstance(out0, int) and not isinstance(out0, float):
            if self.use_pr_conv:
                out0 = self.pr_conv(out0)
            out0 = self.normpr(out0)
            
            #The code to use the skip connection and dropout
            if not self.skip_con_dense:
                out_so = out * x1 * out0
            elif self.skip_con_dense and self.training and 0 < np.random.rand() < self.dropout: 
                out_so = out * x1
            elif not self.training:
                out_so = self.theta * out * x1 * out0 + out * x1
            else:
                out_so = self.beta * self.theta * out * x1 * out0 + out * x1
        else:
            # # if out0 remained int (ie. due to dropout or len(second_ords) ==0).
            out_so = out * x1

        if self.append_bef_norm and not self.use_full_prev_poly:
            second_ords.append(out_so + 0)
        if not self.append_bef_norm and not self.use_full_prev_poly:
            second_ords.append(out_so + 0)
        out_so = self.apply_local_convs(out_so, self.n_lconvs, key='l')
        if self.use_alpha:
            out1 += self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out_so
        if self.use_full_prev_poly:
            second_ords.append(out1 + 0)
        out = self.activ(out1)

        return out, second_ords

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


class D_PolyNets(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None,
                 pool_adapt=False, n_channels=[64, 128, 256, 512], dropout=0, ch_in=3, **kwargs):
        super(D_PolyNets, self).__init__()
        
        # define dropblock and max pooling
        self.dropblock = DropBlock2D(block_size=7, drop_prob=0.1)
        self.maxpool = nn.MaxPool2d(3, 1, 1) 
        print(n_channels)
        self.in_planes = n_channels[0]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
        if not isinstance(dropout, list):
            dropout = [dropout] * len(n_channels)
        self.block_id = 0

        self.conv1 = nn.Conv2d(ch_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, norm_layer=norm_layer[0], dropout=dropout[0], **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, norm_layer=norm_layer[1], dropout=dropout[1], **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, norm_layer=norm_layer[2], dropout=dropout[2], **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, norm_layer=norm_layer[3], dropout=dropout[3], **kwargs)
        self.linear = nn.Linear(n_channels[-1] * block.expansion, num_classes)

        # 16/N zero-mean gaussian initialization    
        for m in self.modules():
            if isinstance(m, BasicBlock):
                print('new initialization!')
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(16 / (m.conv1.weight.shape[1] * np.prod(m.conv1.weight.shape[2:]))))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(16 / (m.conv2.weight.shape[1] * np.prod(m.conv2.weight.shape[2:]))))
                nn.init.normal_(m.lconv0.weight, mean=0, std=np.sqrt(16 / (m.lconv0.weight.shape[1] * np.prod(m.lconv0.weight.shape[2:]))))
                                                 
    def _make_layer(self, block, planes, num_blocks, stride, norm_layer, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                                norm_layer=norm_layer, block_id=self.block_id, **kwargs))
            self.in_planes = planes * block.expansion
            self.block_id += 1
        # # cheeky way to get the activation from the layer1, e.g. in no activation case.
        self.activ = layers[0].activ
        return nn.Sequential(*layers)

    def forward(self, x):
        second_ords = []
        out = self.activ(self.bn1(self.conv1(x)))
        out, second_ords = self.layer1([out, second_ords])
        out, second_ords = self.layer2([self.dropblock(self.maxpool(out)), second_ords])       
        out, second_ords = self.layer3([self.dropblock(self.maxpool(out)), second_ords])        
        out, second_ords = self.layer4([self.maxpool(out), second_ords])
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def D_PolyNets_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return D_PolyNets(BasicBlock, num_blocks, **kwargs)

def test():
    net = D_PolyNets_wrapper()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
