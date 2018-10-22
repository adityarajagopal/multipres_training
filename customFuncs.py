import math 
import sys
import cupy as cp 
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module 
from torch.nn.parameter import Parameter 
from torch.utils.dlpack import to_dlpack, from_dlpack
from customFuncsCuda import cuda_correlate

def correlate(ip, weight, padding, stride, kernel_size, op_size=None, flip=False, dim_switch=False) : 
    if flip :
        w_transform = torch.flip(weight, torch.arange(weight.dim()).tolist())

    if dim_switch : 
        ip = torch.transpose(ip, 1, 0)
        # calculate padding 
        tmp = ((op_size-1)*stride+kernel_size-ip.size(3))
        padding = math.floor(max(tmp, 0)/2)
    
    if flip or dim_switch : 
        weight = torch.transpose(weight, 1, 0)

    o_channels = weight.size(0)
    ip_unfold = torch.nn.functional.unfold(ip, (kernel_size, kernel_size), padding=padding, stride=stride)
    w_unfold = weight.contiguous().view(o_channels, -1) 
    
    # print("op_size :", op_size)
    # print("stride : ", stride)
    # print("padding :", padding)
    # print("kernel_size :", kernel_size)
    # print("o_channels :", o_channels)
    # print("input :", ip.size())
    # print("input_unfold :", ip_unfold.size())
    # print("weight :", weight.size())
    # print("weight_unfold :", w_unfold.size())
    # print()

    # cupy conversion 
    ip_cupy = cp.fromDlpack(to_dlpack(ip_unfold)).astype('int8')
    w_cupy = cp.fromDlpack(to_dlpack(w_unfold)).astype('int8')

    # multiplication in cupy 
    op = cp.matmul(cp.transpose(ip_cupy, (0,2,1)), cp.transpose(w_cupy))
    op = cp.transpose(op, (0,2,1))

    return op 

class Conv2dFunc(Function) : 
    @staticmethod
    def forward(context, ip, weight, bias, stride, padding, dilation, kernel) : 
        kernel = weight.size(2)
        # op = correlate(ip, weight, padding, stride, kernel)        
        op = cuda_correlate(ip, weight, padding, stride, kernel)        
        
        bias_repeat = bias.repeat(op.shape[0], op.shape[2]).view(op.shape[0], op.shape[2], bias.shape[0])
        bias_cupy = cp.fromDlpack(to_dlpack(bias_repeat)).astype('int8')
        op = cp.add(cp.transpose(op, (0,2,1)), bias_cupy) 
        op = cp.transpose(op, (0,2,1))
        
        # conversion back to pytorch 
        op_dlpack = op.toDlpack()
        op_tensor = from_dlpack(op_dlpack).float()

        # fold op tensor 
        h_in = ip.size()[2]
        w_in = ip.size()[3]
        o_dim_x = math.floor((((h_in + 2*padding) - (dilation * (kernel - 1)) - 1)/(stride))+1)
        o_dim_y = math.floor((((w_in + 2*padding) - (dilation * (kernel - 1)) - 1)/(stride))+1)

        op_dim = (o_dim_x, o_dim_y)
        op_fold = torch.nn.functional.fold(op_tensor, op_dim, (1,1))

        context.save_for_backward(ip, weight, bias, Variable(torch.Tensor([padding]).type(torch.cuda.IntTensor), requires_grad=False), Variable(torch.tensor([stride]).type(torch.cuda.IntTensor), requires_grad=False), Variable(torch.Tensor([kernel]).type(torch.cuda.IntTensor)))
        
        return torch.as_tensor(op_fold)

    @staticmethod 
    def backward(context, grad_wrt_op) : 
        # retreive saved tensors 
        ip, weight, bias, padding, stride, kernel = context.saved_tensors 
        padding = int(padding.data.tolist()[0])
        stride = int(stride.data.tolist()[0])
        kernel = int(kernel.data.tolist()[0])

        # grad wrt input matrix 
        if ip.size(2) != 32 : 
            grad_wrt_input = correlate(grad_wrt_op, weight, padding, stride, kernel, flip=True)
            grad_wrt_input_tensor = from_dlpack(grad_wrt_input.toDlpack()).float()
            grad_wrt_input_tensor = torch.nn.functional.fold(grad_wrt_input_tensor, (ip.size()[2], ip.size()[3]), (1,1))
        else : 
            grad_wrt_input_tensor = torch.zeros([128,3,32,32])

        # grad wrt weight matrix 
        op_size = kernel
        kernel = grad_wrt_op.size(2)
        grad_wrt_weight = correlate(ip, grad_wrt_op, padding, stride, kernel, op_size, dim_switch=True) 
        grad_wrt_weight_tensor = from_dlpack(grad_wrt_weight.toDlpack()).float()
        grad_wrt_weight_tensor = torch.nn.functional.fold(grad_wrt_weight_tensor, (weight.size()[2], weight.size()[3]), (1,1))
        grad_wrt_weight_tensor.transpose_(1,0)
        # print(grad_wrt_weight_tensor.size())
        
        # grad wrt bias 
        grad_wrt_bias_tensor = torch.sum(grad_wrt_op, (0,2,3))
        # print(grad_wrt_bias_tensor.size())
        # print()

        return torch.as_tensor(grad_wrt_input_tensor), torch.as_tensor(grad_wrt_weight_tensor), torch.as_tensor(grad_wrt_bias_tensor), None, None, None, None

class Conv2d(Module) : 
    def __init__(self, i_channels, o_channels, kernel_size, stride=1, padding=0) :
        super(Conv2d, self).__init__()
        
        self.i_channels = i_channels 
        self.o_channels = o_channels 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding 
        self.dilation = 1

        # self.weight = Parameter(torch.randint(-128, 127, (o_channels, i_channels, kernel_size, kernel_size), dtype=torch.int8))
        # self.bias = Parameter(torch.randint(-128, 127, (o_channels), dtype=torch.int8))
        self.weight = Parameter(torch.randn((o_channels, i_channels, kernel_size, kernel_size)))
        self.bias = Parameter(torch.randn((o_channels)))

    def forward(self, ip) : 
        return Conv2dFunc.apply(ip, self.weight, self.bias, *(self.stride, self.padding, 1, self.kernel_size))


