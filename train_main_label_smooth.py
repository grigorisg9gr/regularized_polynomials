# Train CIFAR10 with pytorch
from __future__ import print_function
import yaml
import sys
import numpy as np
from time import strftime, time
import random
from os.path import abspath, dirname, join, exists, isdir
from os import curdir, makedirs
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.create_logger import create_logger
from utils import (load_checkpoints, save_checkpoints, create_result_dir, 
                   load_model, print_params, init_weights)

from torchvision_db import return_loaders
try:
    from research_pyutils import mkdir_p, export_pickle
except ImportError:
    from utils.pyutils import mkdir_p, export_pickle

torch.backends.cudnn.benchmark = True
base = dirname(abspath(__file__))
sys.path.append(base)


class LabelSmoothingCrossEntropy(nn.Module):
    """ Perform Label Smoothing for the training"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def train(train_loader, net, optimizer, criterion, train_info, epoch, device):
    """ Perform single epoch of the training."""        
    net.train()
    
    # # initialize variables that are augmented in every batch.
    train_loss, correct, total = 0, 0, 0
    start_time = time()
    for idx, data_dict in enumerate(train_loader):        
        img = data_dict[0]
        label = data_dict[1]

        inputs, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(inputs)
        loss = criterion(pred, label)
        assert not torch.isnan(loss), 'NaN loss.'
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        total += label.size(0)
        correct += predicted.eq(label).cpu().sum()
        if idx % train_info['display_interval'] == 0:
            diff_time = time() - start_time
            acc = float(correct) / total
            m2 = ('Time: {:.04f}, Epoch: {}, Epoch iters: {} / {}\t'
                  'Loss: {:.04f}, Acc: {:.06f}')
            print(m2.format(diff_time, epoch, idx, len(train_loader),
                            float(train_loss), acc))
            logging.info(m2.format(diff_time, epoch, idx, len(train_loader),
                         float(loss.item()), acc))
            start_time = time()
        
    return net


def test(net, test_loader, epoch, device='cuda'):
    """ Perform testing, i.e. run net on test_loader data
        and return the accuracy. """
    cm_predict = []
    cm_target = []
    
    net.eval()
    correct, total = 0, 0
    if hasattr(net, 'is_training'):
        net.is_training = False
    for (idx, data) in enumerate(test_loader):
        sys.stdout.write('\r [%d/%d]' % (idx + 1, len(test_loader)))
        sys.stdout.flush()
        
        img = data[0].to(device)
        label = data[1].to(device)
        
        with torch.no_grad():
             pred = net(img)
        _, predicted = pred.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        cm_predict.append(predicted)
        cm_target.append(label)
    if hasattr(net, 'is_training'):
        net.is_training = True
    return correct / total, torch.cat(cm_predict).view(-1), torch.cat(cm_target).view(-1)


def main(yml_name=None, seed=None, label='', use_cuda=True):
    # # set the seed for all.
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # # set the cuda availability.
    cuda = torch.cuda.is_available() and use_cuda
    device = torch.device('cuda' if cuda else 'cpu')
    # # define whether the arguments will be parsed from the terminal or yml.
    if yml_name is None:
        import argparse
        parser = argparse.ArgumentParser(description='PyTorch cifar10')
        parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
        parser.add_argument('--config_path', type=str, default='configs/resnet_pi-net.yml')
        parser.add_argument('--label', type=str, default='synth')
        args = parser.parse_args()
        yml_name, label = args.config_path, args.label
    #yml = yaml.load(open(yml_name))
    yml = yaml.safe_load(open(yml_name))
    cur_path = abspath(curdir)
    print('Current path: {}.'.format(cur_path))

    if len(label) > 1:
        mainf = '{}_{}_{}'.format(strftime('%Y_%m_%d__%H_%M_%S'), yml['model']['name'], label)
        out = join(cur_path, 'results_resnet', mainf, '')
        # # copy the train files to result dir.
        create_result_dir(out, yml_name, yml)
    else:
        out = join('/tmp', '')

    print('loading data..')
    # # set the dataset options.
    train_loader, test_loader = return_loaders(**yml['dataset'])

    m1 = 'Length of iters per epoch: {}. Length of testing batches: {}.'
    print(m1.format(len(train_loader), len(test_loader)))
    # # load the model.
    modc = yml['model']
    net = load_model(modc['fn'], modc['name'], modc['args']).to(device)
    if torch.cuda.device_count() > 1:
        print('Found {} GPUs.'.format(torch.cuda.device_count()))
        net = nn.DataParallel(net)
    if 'init' in modc['args'].keys():
        net.init = modc['args']['init']
    init_weights(net)
    
    # # define the appropriate paths.
    model_file = mkdir_p(join(out, 'models'))
    logger = create_logger(join(out, 'logs'))
    
    # # define the criterion and the optimizer.
    smoothing = yml['training_info']['smoothing']
    print('the alpha of label smoothing:')
    print(smoothing)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing).to(device)
    params = list(net.parameters())
    sub_params = [p for p in params if p.requires_grad]
    decay = yml['training_info']['weight_dec'] if 'weight_dec' in yml['training_info'].keys() else 5e-4
    optimizer = optim.SGD(sub_params, lr=yml['learning_rate'],
                          momentum=0.9, weight_decay=decay)
    total_params = print_params(net, logging=logging)
    print('The value of weight decay:')
    print(decay)
    net, optimizer, start_epoch = load_checkpoints(net, optimizer, model_file)
    # # get the milestones/gamma for the optimizer.
    tinfo = yml['training_info']
    mil = tinfo['lr_milestones'] if 'lr_milestones' in tinfo.keys() else [40, 60, 80, 100]
    gamma = tinfo['lr_gamma'] if 'lr_gamma' in tinfo.keys() else 0.1
    
    if tinfo['multi_step']:
        print('multi step!')
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mil, gamma=gamma, last_epoch=start_epoch)  
    elif tinfo['exponential_step']:
        print('exponential step!')
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92, last_epoch=start_epoch)
    elif tinfo['cosine_step']:
        print('Cosine step!') 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    best_acc, best_epoch, accuracies = 0, 0, []
    total_epochs = tinfo['total_epochs']

    for epoch in range(start_epoch + 1, total_epochs + 1):
        #scheduler.step()
        net = train(train_loader, net, optimizer, criterion, yml['training_info'], 
                    epoch, device)
        save_checkpoints(net, optimizer, epoch, model_file)
        # # testing mode to evaluate accuracy. 
        acc, predicted, labels = test(net, test_loader, epoch, device=device)
        if acc > best_acc:
            out_path = join(model_file, 'net_best_1.pth')
            state = {'net': net.state_dict(), 'acc': acc, 
                     'epoch': epoch, 'n_params': total_params}
            torch.save(state, out_path)
            best_acc = acc
            best_epoch = epoch

        accuracies.append(float(acc))
        msg = 'Epoch:{}.\tAcc: {:.03f}.\t Best_Acc:{:.03f} (epoch: {}).'
        print(msg.format(epoch,  acc, best_acc, best_epoch))
        logging.info(msg.format(epoch, acc, best_acc, best_epoch))

        scheduler.step()
    d1 = {'acc': accuracies, 'best_acc': best_acc, 'epoch': best_epoch}
    export_pickle(d1, join(out, 'metadata.pkl'))
    
    #test_acc = test(net, test_loader, device=device)
    #logging.info('Test accuracy:{}'.format(test_acc))
    #print('Test accuracy:{}'.format(test_acc))
if __name__ == '__main__':
    main()


