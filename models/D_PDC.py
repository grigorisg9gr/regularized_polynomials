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

from src.iterative_normalization import IterNorm
from src.normalisation_scheme import SwitchNorm2d, DBN, IBN, BatchInstanceNorm2d, DropBlock2D_pol
from dropblock import DropBlock2D, LinearScheduler
import numpy as np


def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif norm_local == 'a':
        norm_local = IBN
    elif norm_local == 2:
        norm_local = IterNorm  
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
                 use_conv_sep=False, div_factor=1, share_norm=False, 
                 append_bef_norm=True, use_so_activ=False, use_pr_conv=False, use_pr_activ=False, 
                 kern_prev=1, norm_pr=-1, dropout=0, train=True, limit_pr=None, sn=False, 
                 sum_prev=False, additive_gn=0, skip_con_dense=False, use_first_dense=True, 
                 use_sep_short=False, block_id=None, use_sep_short_all=False, use_xlconv=True, 
                 use_full_prev_poly=False, use_theta=False, random_scaling=False, in_planes_ho=0, start_block=0, **kwargs):
        """
        D-PDC block. The difference from Pi-net residual block, is that in each block we
        also perform one dense connection from previous blocks. That means that during forward, we pass two inputs.
        :param use_activ: bool; if True, use activation functions in the block.
        :param use_alpha: bool; if True, use a learnable parameter to regularize each previous polynomial in the current Hadamard product.
        :param n_lconvs: int; the number of convolutional layers for the higher order term.
        :param norm_local: int; the type of normalization scheme for the higher order term.
        :param kern_loc:
        :param norm_layer: int; the type of normalization scheme for the first order term.
        :param use_lactiv: bool; if True, use activation functions for the higher order term and 'x' (shortcut one).
        :param norm_x: int; the type of normalization scheme for the 'x' (shortcut one).
        :param n_xconvs:
        :param what_lactiv: int; the type of the activation function for the higher order term.
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
        :param use_so_activ: bool; if True, use activation functions after the multiplication.
        :param use_pr_conv: bool; if True, use convolutions in the dense representations.
        :param use_pr_activ: bool; if True, use an activation in the dense representations.
        :param kern_prev:
        :param norm_pr: int; the type of normalization scheme for the dense representations.
        :param dropout: float; if positive, it uses that as a probability threshold to skip a dense connection
            multiplication.
        :param limit_pr: int; the number of previous Hadamard connections to include; if None, it
            multiplies with all of them.
        :param sn: bool; if True use spectral normalization on the weights.
        :param sum_prev: bool; if True use a skip in the dense product.
        :param additive_gn: If 0, there is no Gaussian noise in the dense connections; if > 0, 
            it samples an Gaussian noise in every iteration for every connection.
        :param kwargs:
        """        
        super(BasicBlock, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_local = get_norm(norm_local)
        self._norm_x = get_norm(norm_x)
        self._norm_bso = get_norm(norm_bso)
        self._norm_prev = get_norm(norm_pr)
        self.use_activ = use_activ
        # # define some 'local' convolutions, i.e. for the second order term only.
        self.n_lconvs = n_lconvs
        self.n_bsoconvs = n_bsoconvs
        self.n_lconvs = n_lconvs
        self.use_lactiv = self.use_activ and use_lactiv

        # for dense connection
        self.append_bef_norm = append_bef_norm
        self.use_so_activ = self.use_activ and use_so_activ
        self.use_pr_conv = use_pr_conv
        self.use_pr_activ = self.use_activ and use_pr_activ and use_pr_conv
        self.is_training = train
        self.dropout = dropout
        self.sn = sn
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

        self.use_uactiv = self.use_activ and use_uactiv
        self.use_only_first_conv = use_only_first_conv
        self.use_conv_sep = use_conv_sep
        self.conv_layer = conv_separ if self.use_conv_sep else nn.Conv2d
        self.div_factor = div_factor
        self.share_norm = share_norm


        self.conv1 = self.conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if norm_layer == 2:
            self.bn1 = IterNorm(planes)
            self._norm_layer = nn.BatchNorm2d        
        elif norm_layer == 'a':
            self.bn1 = IBN(planes, ratio=0.8)
            self._norm_layer = nn.BatchNorm2d
        else:
            self.bn1 = self._norm_layer(planes)

        if sn:
            nn.utils.spectral_norm(self.conv1)    

        if not self.use_only_first_conv:
            self.conv2 = self.conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = self._norm_layer(planes)
            if sn:
                nn.utils.spectral_norm(self.conv2)

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
        self.normpr = self._norm_prev(planes_ho)
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
        self.so_activ = partial(ac1) if self.use_so_activ else lambda x: x
        self.pr_activ = partial(nn.ReLU(inplace=True)) if self.use_pr_activ else lambda x: x
        # # check the output planes for higher-order terms.
        if planes_ho is None or planes_ho < 0:
            planes_ho = planes1
        print('(Internal) planes in the higher-order terms: {}.'.format(planes_ho))

        # for dense connection
        self.cond_short = self.use_sep_short and self.block_id > 0 and self.use_sep_short_all
        print(stride, ' ', block_id, ' !!!!!!!!!!!!!!!!!!!!!!!!!!!: ', in_planes_ho, ' ', planes_ho)
        if (stride != 1 or in_planes_ho != planes_ho) and self.use_sep_short:
            print(stride, ' ', block_id, ' !!!!!!!!!!!!!!!!!!!!!!!!!!!: ', in_planes_ho)
            shortcut = lambda: nn.Sequential(
                    nn.Conv2d(in_planes_ho, planes_ho, kernel_size=1, stride=stride, bias=False),
                    self._norm_layer(planes_ho)
                )
            #self.cond_short = self.use_sep_short and self.block_id > 0 and self.use_sep_short_all
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
        
        if self.use_pr_conv:
            self.pr_conv = nn.Conv2d(planes_ho, planes_ho, kernel_size=kern_prev, stride=1, padding=int(kern_prev > 1))
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

        self.use_theta = use_theta
        self.random_scaling = random_scaling
        print('For dense connection: ')
        print('block id: {}'.format(self.block_id))
        print('norm layer: {}'.format(norm_layer))
        print('norm_x: {}'.format(norm_x))
        print('norm_local: {}'.format(norm_local))
        print('norm_prev: {}'.format(norm_pr))
        print('use active: {}'.format(self.use_activ))
        print('use lconv active: {}'.format(self.use_lactiv))
        print('use so active: {}'.format(self.use_so_activ))
        print('use pr active: {}'.format(self.use_pr_activ))
        print('drop out: {}'.format(self.dropout))
        print('use xconv: {}'.format(self.use_xlconv))
        print('use pr conv: {}'.format(self.use_pr_conv))
        print('use sep short: {}'.format(self.use_sep_short))
        print('use sep short all: {}'.format(self.use_sep_short_all))
        print('sum prev: {}'.format(self.sum_prev))
        print('add gaussian: {}'.format(self.additive_gn))
        print('skip dense connection: {}'.format(self.skip_con_dense))
        print('use full polynomials: {}'.format(self.use_full_prev_poly))
        print('use sn: {}'.format(self.sn))
        print('use theta: {}'.format(self.use_theta))
        print('use alpha: {}'.format(self.use_alpha))
        print('use random scaling: {}'.format(self.random_scaling))
        print('use limit: {}'.format(self.limit_prev))


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
        if not self.use_only_first_conv:
            out = self.bn2(self.conv2(out))
        out1 = out + self.shortcut(x)
        # # normalize the x (shortcut one).
        x1 = self.normx(self.shortcut(x))
        x1 = self.apply_local_convs(x1, self.n_bsoconvs, key='bso')
        x2 = self.apply_convs_so(x1, key=1)
        x3 = self.apply_convs_so(x1, key=2)
        
        # # for dense connection
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

        if len(second_ords) < self.limit_prev:
            for cnt, prev_so in enumerate(second_ords[::-1][self.use_fd:]):
                # if self.training and 0 < np.random.rand() < self.dropout: 
                #     continue
                if self.additive_gn > 0 and self.training:
                    prev_so = prev_so + self.additive_gn * torch.randn(*prev_so.shape).to(x.device)
                if cnt == 0:
                    out0 = prev_so
                elif self.sum_prev:
                    out0 = out0 + out0 * prev_so                   
                else:
                    #print(out0.size(), prev_so.size())
                    out0 = out0 * prev_so
 
        else:
            for cnt, prev_so in enumerate(second_ords[::-1][self.use_fd: self.limit_prev]):
                # if self.training and 0 < np.random.rand() < self.dropout:
                #     continue
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
            out0 = self.pr_activ(self.normpr(out0)) 
            
            #The code to use the skip connection and dropout
            if not self.skip_con_dense:
                out_so = x2 * x3  * out0
            elif self.skip_con_dense and self.training and 0 < np.random.rand() < self.dropout: 
                out_so = x2 * x3 
            elif not self.training:
                out_so = self.theta * x2 * x3  * out0 + x2 * x3 
            else:
                #print(x2.size(), x3.size(), out0.size())
                out_so = self.beta * self.theta * x2 * x3  * out0 + x2 * x3 
        else:
            # # if out0 remained int (ie. due to dropout or len(second_ords) ==0).
            out_so = x2 * x3
        out_so = self.so_activ(out_so)      
        #####################################################################################################

        if self.append_bef_norm and not self.use_full_prev_poly:
            second_ords.append(out_so + 0)
        if self.div_factor != 1:
            out_so = out_so * 1. / self.div_factor
        out_so = self.apply_local_convs(out_so, self.n_lconvs, key='l')
        if not self.append_bef_norm and not self.use_full_prev_poly:
            second_ords.append(out_so + 0)
        if self.use_alpha:
            out1 += self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 += out_so
        if self.use_full_prev_poly:
            second_ords.append(out1 + 0)
        out = self.activ(out1)
        return out, second_ords

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


class D_PDC(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None, out_activ=False, pool_adapt=False, 
                 n_channels=[64, 128, 256, 512], planes_ho=None, dropout=0, ch_in=3, **kwargs):
        super(D_PDC, self).__init__()

        # define dropblock and max pooling
        self.dropblock = DropBlock2D(block_size=7, drop_prob=0.1)
        self.maxpool = nn.MaxPool2d(3, 1, 1)
        
        # planes ho
        self.in_planes_ho = planes_ho[0]

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

        if not isinstance(dropout, list):
            dropout = [dropout] * len(n_channels)
        self.block_id = 0

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

        # 16/N zero-mean gaussian initialization   
        for m in self.modules():
            if isinstance(m, BasicBlock):
                print('new initialization!')
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(16 / (m.conv1.weight.shape[1] * np.prod(m.conv1.weight.shape[2:]))))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(16 / (m.conv2.weight.shape[1] * np.prod(m.conv2.weight.shape[2:]))))
                nn.init.normal_(m.lconv0.weight, mean=0, std=np.sqrt(16 / (m.lconv0.weight.shape[1] * np.prod(m.lconv0.weight.shape[2:]))))


    def _make_layer(self, block, planes, num_blocks, stride, planes_ho, norm_layer, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, 
                                norm_layer=norm_layer, block_id=self.block_id, planes_ho=planes_ho, in_planes_ho=self.in_planes_ho, **kwargs))
            self.in_planes = planes * block.expansion
            self.in_planes_ho = planes_ho * block.expansion
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
        if len(self.n_channels) > 4:
            for i in range(len(self.n_channels) - 4):
                out = getattr(self, 'layer{}'.format(i + 5))(out)
        if self.out_activ:
            out = self.oactiv(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

def D_PDC_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return D_PDC(BasicBlock, num_blocks, **kwargs)


def test():
    net = D_PDC_wrapper()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
