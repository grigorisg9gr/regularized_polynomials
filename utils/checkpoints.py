from os.path import isdir, join, exists
import torch


def load_checkpoints(net, optimizer, model_path):
    print(model_path)
    latestpath = join(model_path, 'latest.pth40.tar')
    last_epoch = -1
    if exists(latestpath):
        print('===================>loading the checkpoints from:', latestpath)
        latest = torch.load(latestpath)
        last_epoch = latest['epoch']
        net.load_state_dict(latest['net'])
        optimizer.load_state_dict(latest['optim'])
    else:
        print('=====================>Train From Scratch')
    return net, optimizer, last_epoch


def save_checkpoints(net, optimizer, epoch, model_path):
    latest = {}
    latest['epoch'] = epoch
    latest['net'] = net.state_dict()
    latest['optim'] = optimizer.state_dict()
    #torch.save(latest, join(model_path, 'latest.pth.tar'))
    if epoch % 20 == 0:
        torch.save(latest, join(model_path, 'latest.pth%d.tar' % epoch))
