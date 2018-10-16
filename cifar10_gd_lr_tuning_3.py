from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
import copy
import csv
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from torch.optim.optimizer import Optimizer, required

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np 

import quantiser
# import customoptim

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize')
# Hyperparameter tuning
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--mo-schedule', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')
parser.add_argument('--mo-list', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')

parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--sf', type=float, default=1.0, metavar='N')
parser.add_argument('--c16', type=int, default=350, metavar='N', help='number of iterations of fp16 to perform')
#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# ----------------------------------------------------- Frequently used arguments ----------------------------------------------------------------------  
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--max-lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--min-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--delta', default=0.04, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=None, help='Decrease learning rate at these epochs.')
parser.add_argument('--lr-schedule', type=float, nargs='+', default=None, help='Fixed values for lr at corresponding schedule epoch.')
parser.add_argument('--prec-schedule', type=float, nargs='+', default=None, help='Epochs at which precision change should happen')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--bw', type=int, default=2, metavar='N', help='quantisation bit width')
parser.add_argument('--quant', type=int, default=1, metavar='N', help='perform quantisation or not')
parser.add_argument('--resolution', type=int, default=3, metavar='N', help='epochs over which gd is calculated')
parser.add_argument('--lr-resolution', type=int, default=5, metavar='N', help='epochs over which gd is calculated')
parser.add_argument('--prec-thresh', type=float, default=1.3, metavar='N', help='drop in gd that causes precision change')
parser.add_argument('--lr-var-thresh', type=float, default=0.03, metavar='N', help='drop in gd that causes precision change')
parser.add_argument('--prec-count-thresh', type=int, default=20, metavar='N', help='number of epochs after which force change precision')
parser.add_argument('--low-prec-limit', type=int, default=8, metavar='N', help='number of epochs after which force change precision')
parser.add_argument('--patience', type=int, default=2, metavar='N', help='number of epochs after which force change precision')
parser.add_argument('-e', '--evaluate', type=str, default=None, metavar='PATH', help='evaluate model on validation set')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
convLayers = [] 
bit_width = args.bw

sum_of_norms = {}
sum_of_grads = {}
def update_grad_div_calc (layer, grad) : 
    global sum_of_norms 
    global sum_of_grads 

    if layer in sum_of_norms : 
        sum_of_norms[layer].add_(torch.pow(torch.norm(grad,2),2))
    else : 
        sum_of_norms[layer] = torch.pow(torch.norm(grad, 2),2)

    if layer in sum_of_grads : 
        sum_of_grads[layer].add_(grad)
    else : 
        sum_of_grads[layer] = grad

def quantise (self, input, output) : 
    global bit_width
    output = quantiser.quantise_inputs(output, bit_width)
    
def register_hooks (arch, depth, model) : 
    if arch == 'alexnet': 
        model.module.conv1.register_forward_hook(quantise)
        model.module.conv2.register_forward_hook(quantise)
        model.module.conv3.register_forward_hook(quantise)
        model.module.conv4.register_forward_hook(quantise)
        model.module.conv5.register_forward_hook(quantise)
        model.module.relu.register_forward_hook(quantise) 
        model.module.classifier.register_forward_hook(quantise)
        
        model.module.conv1.register_backward_hook(quantise)
        model.module.conv2.register_backward_hook(quantise)
        model.module.conv3.register_backward_hook(quantise)
        model.module.conv4.register_backward_hook(quantise)
        model.module.conv5.register_backward_hook(quantise)
        model.module.relu.register_backward_hook(quantise) 
        model.module.classifier.register_backward_hook(quantise)
    
    if arch == 'resnet' and depth  == 20 : 
        model.module.conv1.register_forward_hook(quantise)
        model.module.relu.register_forward_hook(quantise)
        model.module.layer1.register_forward_hook(quantise)
        model.module.layer2.register_forward_hook(quantise)
        model.module.layer3.register_forward_hook(quantise)
        model.module.avgpool.register_forward_hook(quantise)
        model.module.fc.register_forward_hook(quantise)
        
        model.module.conv1.register_backward_hook(quantise)
        model.module.relu.register_backward_hook(quantise)
        model.module.layer1.register_backward_hook(quantise)
        model.module.layer2.register_backward_hook(quantise)
        model.module.layer3.register_backward_hook(quantise)
        model.module.avgpool.register_backward_hook(quantise)
        model.module.fc.register_backward_hook(quantise)

    if arch == 'vgg16_bn' : 
        model.module.conv1.register_forward_hook(quantise)
        model.module.conv2.register_forward_hook(quantise)
        model.module.conv3.register_forward_hook(quantise)
        model.module.conv4.register_forward_hook(quantise)
        model.module.conv5.register_forward_hook(quantise)
        model.module.conv6.register_forward_hook(quantise)
        model.module.conv7.register_forward_hook(quantise)
        model.module.conv8.register_forward_hook(quantise)
        model.module.conv9.register_forward_hook(quantise)
        model.module.conv10.register_forward_hook(quantise)
        model.module.conv11.register_forward_hook(quantise)
        model.module.conv12.register_forward_hook(quantise)
        model.module.conv13.register_forward_hook(quantise)
        model.module.relu.register_forward_hook(quantise)
        model.module.maxpool.register_forward_hook(quantise)
        
        model.module.conv1.register_backward_hook(quantise)
        model.module.conv2.register_backward_hook(quantise)
        model.module.conv3.register_backward_hook(quantise)
        model.module.conv4.register_backward_hook(quantise)
        model.module.conv5.register_backward_hook(quantise)
        model.module.conv6.register_backward_hook(quantise)
        model.module.conv7.register_backward_hook(quantise)
        model.module.conv8.register_backward_hook(quantise)
        model.module.conv9.register_backward_hook(quantise)
        model.module.conv10.register_backward_hook(quantise)
        model.module.conv11.register_backward_hook(quantise)
        model.module.conv12.register_backward_hook(quantise)
        model.module.conv13.register_backward_hook(quantise)
        model.module.relu.register_backward_hook(quantise)
        model.module.maxpool.register_backward_hook(quantise)


patience = args.patience
gd_violations = 0
lr_violations = []
exp_threshold = [1 + 1.5*math.exp(-0.1*x) for x in range(args.epochs)]
memory = 0
memory_counter = False
prec_count = 0
end_training = False
def update_hyperparameters (optimizer, epoch, metaParamDict, loss, mean_gd, max_gd, loss_sd, loss_mean, pchange_loss):
    global state
    global prev_loss
    global bit_width
    global patience
    global gd_violations
    global lr_violations
    global exp_threshold
    global prec_count
    global memory 
    global memory_counter
    global lr_ub 
    global lr_lb 
    global end_training

    print (prec_count, bit_width)
    
    # update learning rate
    if args.schedule is not None : 
        if epoch in args.schedule:
            if args.lr_schedule == None : 
                state['lr'] *= args.gamma
            else : 
                new_lr = args.lr_schedule[args.schedule.index(epoch)]
                state['lr'] = new_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['lr']

    else :
        if bit_width == 32 and prec_count == 15 : 
            if state['lr'] > args.min_lr : 
                state['lr'] *= args.gamma 
                for param_group in optimizer.param_groups:
                    param_group['lr'] = state['lr']
                prec_count = -1
            else : 
                end_training = True
                
    # update precision change
    prev_prec = bit_width
    if args.prec_schedule is not None : 
        if epoch in args.prec_schedule : 
            if bit_width < args.low_prec_limit: 
                bit_width += 2
            else : 
                if metaParamDict['quantised'] == True : 
                    metaParamDict['quantised'] = False 
                    bit_width = 32
    
    else : 
        if (epoch % args.resolution) == 0 : 
            if mean_gd >= max_gd :
                max_gd = mean_gd
            
            else : 
                if (max_gd / mean_gd) > exp_threshold[epoch] : 
                    print ('gd_violation: ', gd_violations, max_gd, mean_gd, exp_threshold[epoch])
                    gd_violations += 1
    
    if gd_violations >= patience : 
        print (gd_violations, bit_width)
        print (mean_gd, max_gd)
        gd_violations = 0
        
        if bit_width < args.low_prec_limit : 
            print ('increasing precision')
            bit_width += 2
            max_gd = 0 
            prec_count = -1
        
        else : 
            if metaParamDict['quantised'] == True : 
                print ('removing quantisation')
                metaParamDict['quantised'] = False
                bit_width = 32
                max_gd = 0
                prec_count = -1
                patience += 1

            # else : 
            #     if bit_width == 32 :
            #         if state['lr'] > args.min_lr : 
            #             state['lr'] *= args.gamma 
            #             for param_group in optimizer.param_groups:
            #                 param_group['lr'] = state['lr']
            #             max_gd = 0
            #             prec_count = -1
            #         else : 
            #             end_training = True
    
    prec_count += 1
    prev_loss = loss
    
    return prev_prec, max_gd

def save_checkpoint (state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def test (testloader, model, criterion, epoch, use_cuda, metaParamDict):
    global best_acc
    global bit_width

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if metaParamDict['quantised'] : 
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
            inputs = quantiser.quantise_inputs(inputs, bit_width)
        else : 
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def train (model, master_copy, criterion, optimizer, inputs, targets, metaParamDict, logParamDict):
    global bit_width
    global convLayers
    # switch to train mode
    model.train()

    logParamDict['data_time'].update(time.time() - logParamDict['end'])

    if metaParamDict['quantised'] : 
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs = quantiser.quantise_inputs(inputs, bit_width)
    else : 
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

    # compute output
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    if metaParamDict['quantised'] :
        loss = loss * args.sf

    # measure accuracy and record loss
    prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
    logParamDict['losses'].update(loss.data[0], inputs.size(0))
    logParamDict['top1'].update(prec1[0], inputs.size(0))
    logParamDict['top5'].update(prec5[0], inputs.size(0))
    
    # compute gradient and do SGD step
    model.zero_grad()
    loss.backward()

    if metaParamDict['quantised'] : 
        # quantiser.quantise_grad(model, convLayers, bit_width)
        for param, param_w_grad in zip(master_copy, list(model.parameters())) : 
            if param.grad is None : 
                param.grad = torch.nn.Parameter(param.data.new().resize_(*param.data.size()))
            param.grad.data.copy_(param_w_grad.grad.data)

        for param in master_copy : 
                param.grad.data = param.grad.data / args.sf

    # update weights
    optimizer.step()

    if metaParamDict['quantised'] :
        params = list(model.parameters()) 
        for i in range(len(params)) : 
            params[i].data.copy_(master_copy[i].data)
        quantiser.quantise_weights(model, convLayers, bit_width) 

    # measure elapsed time
    logParamDict['batch_time'].update(time.time() - logParamDict['end'])
    logParamDict['end'] = time.time()

    # plot progress
    if logParamDict['printBar'] :
        logParamDict['bar'].suffix  = '({batch}/{size}) ({pres:}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=metaParamDict['batch_idx'] + 1,
                    size=metaParamDict['size'],
                    # pres='16' if metaParamDict['quantised'] else '32',
                    pres= bit_width if metaParamDict['quantised'] else '32',
                    data=logParamDict['data_time'].avg,
                    bt=logParamDict['batch_time'].avg,
                    total=logParamDict['bar'].elapsed_td,
                    eta=logParamDict['bar'].eta_td,
                    loss=logParamDict['losses'].avg,
                    top1=logParamDict['top1'].avg,
                    top5=logParamDict['top5'].avg,
                    )
        logParamDict['bar'].next()

def main ():
    global best_acc
    global convLayers
    global sum_of_norms 
    global sum_of_grads
    global bit_width
    global lr_violations 
    global gd_violations
    global memory 
    global memory_counter
    global end_training 
    
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    
    # setup dual copy of models
    model_quant = copy.deepcopy(model)
    model = torch.nn.DataParallel(model).cuda()
    model_quant = torch.nn.DataParallel(model_quant).cuda()

    convLayers = quantiser.calculate_layers_to_quantise(model)
    
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    # optimizer = customoptim.CustomSGDMultiPres(list(model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    metaParamDict = {}

    max_gd = 0
    loss_list = []
    loss_sd = 0
    loss_mean = 0
    prev_loss = 10
    pchange_loss = 0
    mean_gd = 0
    metaParamDict['quantised'] = (args.quant == 1)
    
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # resume logging state 
        prev_loss = checkpoint['prev_loss'] 
        loss_sd = checkpoint['loss_sd'] 
        loss_mean = checkpoint['loss_mean'] 
        loss_list = checkpoint['loss_list'] 
        pchange_loss = checkpoint['pchange_loss']
        sum_of_norms = checkpoint['s_o_n'] 
        sum_of_grads = checkpoint['s_o_g'] 
        max_gd = checkpoint['max_gd'] 
        mean_gd = checkpoint['mean_gd']
        lr_violations = checkpoint['lr_violations'] 
        memory = checkpoint['memory'] 
        memory_counter = checkpoint['memory_counter'] 
        gd_violations = checkpoint['gd_violations'] 
        # bit_width = checkpoint['bit_width'] 
        bit_width = 32
        if bit_width == 32 : 
            metaParamDict['quantised'] = False 
        elif args.quant == 1 :
            metaParamDict['quantised'] = True

        # tmp = args.checkpoint.split('/')
        # checkpoint_parent_dir = ''
        # for i in range(tmp.index("checkpoints")) : 
        #     checkpoint_parent_dir = os.path.join(checkpoint_parent_dir, tmp[i])
        # logger = Logger(os.path.join(checkpoint_parent_dir, 'log.txt'), title=title, resume=True)
        # grad_logger = Logger(os.path.join(checkpoint_parent_dir, 'grad_log.txt'), title=title, resume=True)
        open(os.path.join(args.checkpoint, 'log.txt'), 'w+')
        open(os.path.join(args.checkpoint, 'grad_log.txt'), 'w+')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger.set_names(['Learning Rate', 'Epoch Precision', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        grad_logger = Logger(os.path.join(args.checkpoint, 'grad_log.txt'), title=title, resume=True)
        grad_logger.set_names(['Epoch', 'Mean GD', 'Max GD', 'Loss SD', 'Loss Mean', 'PChange Loss', 'LR Violations', 'GD Violations'])
    
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Epoch Precision', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        
        grad_logger = Logger(os.path.join(args.checkpoint, 'grad_log.txt'), title=title)
        grad_logger.set_names(['Epoch', 'Mean GD', 'Max GD', 'Loss SD', 'Loss Mean', 'PChange Loss', 'LR Violations', 'GD Violations'])
    
    logParamDict = {} 

    if args.evaluate is not None:
        print('\nEvaluation only')
        metaParamDict['quantised'] = False
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda, metaParamDict)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    prev_prec = bit_width
    # register_hooks(args.arch, args.depth, model_quant)
    train_loss = 0 
    for epoch in range(start_epoch, args.epochs):
        prev_prec, max_gd = update_hyperparameters(optimizer, epoch, metaParamDict, train_loss, mean_gd, max_gd, loss_sd, loss_mean, pchange_loss)

        checkpoint_file = os.path.join(args.checkpoint, 'checkpoints', 'epoch_'+str(epoch+1))
        grad_dir = os.path.join(args.checkpoint, 'grads')

        if not os.path.isdir(checkpoint_file):
            mkdir_p(checkpoint_file)
        if not os.path.isdir(grad_dir):
            mkdir_p(grad_dir)
        
        metaParamDict['epoch'] = epoch 
        metaParamDict['use_cuda'] = use_cuda
        metaParamDict['size'] = len(trainloader)
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        logParamDict['batch_time'] = batch_time
        logParamDict['data_time'] = data_time 
        logParamDict['losses'] = losses 
        logParamDict['top1'] = top1 
        logParamDict['top5'] = top5 
        logParamDict['end']  = end
        logParamDict['printBar']  = True
        
        logParamDict['losses_tmp'] = []
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        
        if logParamDict['printBar'] : 
            logParamDict['bar'] = Bar('Processing', max=len(trainloader), redirect_stdout=True)

        log_model = None
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            metaParamDict['batch_idx'] = batch_idx
            
            if metaParamDict['quantised'] : 
                quantiser.quantise_weights(model_quant, convLayers, bit_width) 
                train(model_quant, list(model.parameters()), criterion, optimizer, inputs, targets, metaParamDict, logParamDict)
                log_model = model_quant
            else : 
                train(model, list(model.parameters()), criterion, optimizer, inputs, targets, metaParamDict, logParamDict)
                log_model = model
            
        # calculate gradient diversities
        for elem in list(log_model.named_parameters()) : 
            update_grad_div_calc(elem[0], elem[1].grad.data)

        if ((epoch+1) % args.resolution) == 0 : 
            mean_gd = 0
            num_gd = 0
            for layer in sum_of_grads : 
                mean_gd += (sum_of_norms[layer] / torch.pow(torch.norm(sum_of_grads[layer],2),2))
                num_gd += 1
            mean_gd = mean_gd / num_gd
            sum_of_norms = {}
            sum_of_grads = {}
        
        # save gradient values to grads file 
        for elem in list(log_model.named_parameters()) : 
            name = elem[0].split('.')
            csv_file = grad_dir+'/'+str(name[1]+'-'+name[2])+'.csv' 
            grad = elem[1].grad.data
            to_print = [epoch]
            to_print += grad.reshape(1,-1).data.tolist()[0]
            with open(csv_file,'a') as log_file : 
                wr = csv.writer(log_file)
                wr.writerows([to_print])
        
        if logParamDict['printBar'] : 
            logParamDict['bar'].finish()
        train_loss = logParamDict['losses'].avg 
        train_acc = logParamDict['top1'].avg

        # calculate variance of precent change in lr
        if epoch == 0 : 
            prev_loss = train_loss
        else : 
            # pchange_loss = torch.cuda.FloatTensor([(prev_loss - train_loss)/prev_loss])
            pchange_loss = (prev_loss - train_loss)/prev_loss
            loss_list.append(pchange_loss)
            prev_loss = train_loss
        
        if len(loss_list) == args.lr_resolution : 
            loss_sd = math.sqrt(np.var(loss_list))
            # loss_mean = torch.cuda.FloatTensor(np.asarray(np.mean(loss_list)))
            loss_mean = np.mean(loss_list)
            loss_list = []
        
        grad_logger.append([epoch, mean_gd, max_gd, loss_sd, loss_mean, pchange_loss, len(lr_violations), gd_violations])
        
        if metaParamDict['quantised'] : 
            test_loss, test_acc = test(testloader, model_quant, criterion, epoch, use_cuda, metaParamDict)
        else : 
            test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, metaParamDict)
    
        # append logger file
        logger.append([state['lr'], bit_width, train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'prev_loss' : prev_loss,
                'loss_sd' : loss_sd, 
                'loss_mean' : loss_mean,
                'loss_list' : loss_list,
                'pchange_loss' : pchange_loss,
                's_o_n' : sum_of_norms, 
                's_o_g' : sum_of_grads, 
                'max_gd' : max_gd,
                'mean_gd' : mean_gd,
                'lr_violations' : lr_violations, 
                'memory' : memory, 
                'memory_counter' : memory_counter,
                'gd_violations' : gd_violations,
                'bit_width' : bit_width
            }, is_best, checkpoint=checkpoint_file)
        
        if end_training == True : 
            break

    grad_logger.close()
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


if __name__ == '__main__':
    main()
