from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

def calculate_layers_to_quantise (model) : 
    count = 0 
    procLayers = [] 
    for param in model.parameters() : 
        if len(list(param.size())) > 2 : 
            if (list(param.size())[2]) > 1 : 
                procLayers.append(count) 
        elif len(list(param.size())) > 1 : 
            procLayers.append(count)
        count += 1
    return procLayers

def ensure_values_lt_1(mat) : 
    count = 0
    while mat.ge(1.0).nonzero().size() != torch.Size([0]) : 
        mat.div_(10.0)
        count = count + 1
    return count

def quantise_weights (model, convLayers, bit_width) : 
    count = 0
    for param in model.parameters() : 
        if count in convLayers : 
            sign_mat = torch.sign(param.data)
            abs_mat = torch.abs(param.data)
            scale_fac = ensure_values_lt_1(abs_mat)
            fxdPt_mat = fixed_single_sf(abs_mat, bit_width)
            fxdPt_mat.mul_(pow(10,scale_fac))
            param.data = fxdPt_mat * sign_mat
        count += 1

def quantise_grad (model, convLayers, bit_width) : 
    count = 0
    for param in model.parameters() : 
        if count in convLayers : 
            sign_mat = torch.sign(param.grad.data)
            abs_mat = torch.abs(param.grad.data)
            scale_fac = ensure_values_lt_1(abs_mat)
            fxdPt_mat = fixed_single_sf(abs_mat, bit_width)
            fxdPt_mat.mul_(pow(10,scale_fac))
            param.grad.data = fxdPt_mat * sign_mat
        count += 1

def quantise_inputs (inputs, bit_width) : 
    if isinstance(inputs, torch.Tensor) : 
        tmp = inputs.clone()
        sign_mat = torch.sign(tmp)
        scale_fac = ensure_values_lt_1(tmp)
        abs_mat = torch.abs(tmp)
        fxdPt_mat = fixed_single_sf(abs_mat, bit_width)
        fxdPt_mat.mul_(pow(10,scale_fac))
        tmp = fxdPt_mat * sign_mat
        return tmp
    else : 
        tmp = inputs[0]
        sign_mat = torch.sign(tmp)
        scale_fac = ensure_values_lt_1(tmp)
        abs_mat = torch.abs(tmp)
        fxdPt_mat = fixed_single_sf(abs_mat, bit_width)
        fxdPt_mat.mul_(pow(10,scale_fac))
        tmp = fxdPt_mat * sign_mat
        return (tmp)

def matAND (matA, matB) : 
    return matA * matB

def matOR (matA, matB):
    return torch.ge(matA + matB, 1).byte()

def fixed_multiple_sf (num) : 
    dim = num.size()
    count = torch.ne(num,0.0).float().cuda()
    cu = torch.cuda.ByteTensor(dim).zero_()
    result = torch.cuda.ByteTensor(dim).zero_()
    sf = torch.zeros(dim).cuda()
    
    num.mul_(2.0)
    binMask = torch.ge(count, 1.0).cuda().byte()
    counter = 0
    while (count.nonzero().size() != torch.Size([])) : 
        counter = counter + 1
        byte = num.byte()
        
        num.sub_(byte.float()).mul_(2.0)
    
        byte = matAND(binMask, byte)
        cu = matOR(cu, byte)
        cu = matAND(binMask, cu)
        
        count.add_(cu.float())
        binMask = torch.ge(count, 1.0).byte()
    
        sf.add_(binMask.float())
    
        count = torch.fmod(count, 10)
        binMask = torch.ge(count, 1.0).byte()
        
        result.add_(matAND(cu, byte).add(result * cu * binMask))
    
    float_fxd = result.float() * torch.pow(2, sf.sub_(1).mul_(-1))
    return float_fxd

def fixed_single_sf (num, bit_width) : 
    dim = num.size()
    counter = 0
    cu = 0
    result = torch.cuda.FloatTensor(dim).zero_()
    sf = 0 
    bits = bit_width

    if (num.nonzero().size() == torch.Size([0])) : 
        return num 

    num.mul_(2.0)
    while (counter < (bits+1)) : 
        byte = num.byte()
        if (torch.max(byte) == 1) and (cu == 0) : 
            cu = 1
        
        num.sub_(byte.float()).mul_(2.0)
    
        sf += 1 
        
        result.add_(result * (counter != bits)).add_(byte.float())
        counter = counter + cu
    
    float_fxd = result.float() * pow(2, -(sf-1))
    return float_fxd
