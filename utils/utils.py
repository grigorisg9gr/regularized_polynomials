'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np

def init_weights(net, gain_ortho=1, gain_xavier=1, mean_norm=0, std_norm=1, **kwargs):
    """
    Initialize the pytorch weights of a network. It uses a single format of initialization
    saved in net.init.
    :param net: pytorch type network; it's weights will be initialized.
    :param gain_ortho: float; the gain in the orthogonal initialization.
    :param gain_xavier: float; the gain in the xavier initialization.
    :param mean_norm: float; the mean in the gaussian initialization.
    :param std_norm: float; the std in the gaussian initialization.
    :return: int; The number of parameters.
    """
    from torch.nn import init
    import torch.nn as nn

    if not hasattr(net, 'init'):
        print('[Utils] No net.init found; returning')
        return
    param_count = 0
    for module in net.modules():
        if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or
                isinstance(module, nn.Embedding) or isinstance(module, nn.ConvTranspose2d)):
            if net.init in ['gaussian', 'normal', 'N', 0]:
                print('gaussian initialization')
                init.normal_(module.weight, mean_norm, std_norm)
            elif net.init in ['glorot', 'xavier', 1]:
                print('xavier')
                print(gain_xavier)
                init.xavier_uniform_(module.weight, gain=gain_xavier)
            elif net.init in ['ortho', 2]:
                print('ortho')
                init.orthogonal_(module.weight, gain=gain_ortho)
            elif net.init in ['kaiming', 'he_normal', 3]:
                print('kaiming normal')
                init.kaiming_normal_(module.weight)
            elif net.init in ['kaimingun', 'he_uniform', 4]:
                print('kaiming gun')
                init.kaiming_uniform_(module.weight)
            else:
                print('Init style not recognized...')
                #special_uniform_(module.weight)
        param_count += sum([p.data.nelement() for p in module.parameters()])
    return param_count

# add
def fixup_init(m, t):
  if t == 1:
    nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * 2 ** (-0.5))
    nn.init.constant_(m.conv2.weight, 0)
    if m.downsample is True:
      nn.init.normal_(m.shortcut[0].weight, mean=0, std=np.sqrt(2 / (m.shortcut[0].weight.shape[0] * np.prod(m.shortcut[0].weight.shape[2:]))))
    
  elif isinstance(m, nn.Linear):
    nn.init.constant_(m.weight, 0)
    nn.init.constant_(m.bias, 0)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def special_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    print('bob')
    fan = calculate_correct_fan(tensor, mode)
    gain = 1.0
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

    

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#try:
#    _, term_width = os.popen('stty size', 'r').read().split()
#    term_width = int(term_width)
#except:
#    pass

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def print_params(net, logging=None):
    """ Print the number of parameters (and the ones with gradients). """
    total_params = sum(p.numel() for p in net.parameters())
    sub_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    m3 = 'total params: {}, sub_params: {}'
    print(m3.format(total_params, sub_params))
    if logging is not None:
        logging.info(m3.format(total_params, sub_params))
    return total_params
