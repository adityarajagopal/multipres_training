import math 
import sys
import cupy as cp 
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module 
from torch.nn.parameter import Parameter 
from torch.utils.dlpack import to_dlpack, from_dlpack

def correlate(ip, weight, padding, stride, kernel_size, corner_case=False, op_channels=None) : 
    if corner_case:
        w_transform = torch.flip(weight, torch.arange(weight.dim()).tolist())
    o_channels = weight.size(0)
    ip_unfold = torch.nn.functional.unfold(ip, (kernel_size, kernel_size), padding=padding, stride=stride)
    w_unfold = weight.view(o_channels, -1) 
    
    # print("kernel_size :", kernel_size)
    # print("o_channels :", o_channels)
    # print("input :", ip.size())
    # print("input_unfold :", ip_unfold.size())
    # print("wegiht :", weight.size())
    print("weight_unfold :", w_unfold.size())

    # cupy conversion 
    ip_cupy = cp.fromDlpack(to_dlpack(ip_unfold)).astype('int8')
    w_cupy = cp.fromDlpack(to_dlpack(w_unfold)).astype('int8')

    # print(ip_cupy.shape)
    # print(w_cupy.shape)

    # multiplication in cupy 
    op = cp.matmul(cp.transpose(ip_cupy, (0,2,1)), cp.transpose(w_cupy))
    op = cp.transpose(op, (0,2,1))
    print(op.shape)


    return op 

class Conv2dFunc(Function) : 
    @staticmethod
    def forward(context, ip, weight, bias, stride, padding, dilation, kernel) : 
        kernel = weight.size(2)
        op = correlate(ip, weight, padding, stride, kernel)        
        
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
        print("output size:\t\t", op_fold.size())

        context.save_for_backward(ip, weight, bias, Variable(torch.Tensor([padding]).type(torch.cuda.IntTensor), requires_grad=False), Variable(torch.tensor([stride]).type(torch.cuda.IntTensor), requires_grad=False), Variable(torch.Tensor([kernel]).type(torch.cuda.IntTensor)))
        
        return torch.as_tensor(op_fold)

    @staticmethod 
    def backward(context, grad_wrt_op) : 
        # retreive saved tensors 
        ip, weight, bias, padding, stride, kernel = context.saved_tensors 
        padding = int(padding.data.tolist()[0])
        stride = int(stride.data.tolist()[0])
        kernel = int(kernel.data.tolist()[0])

        # grad wrt weight matrix 
        # print(ip.shape)
        grad_wrt_weight = correlate(grad_wrt_op, weight, padding, stride, kernel, True, weight.size(0))
        grad_wrt_weight_tensor = from_dlpack(grad_wrt_weight.toDlpack()).float()
        grad_wrt_weight_tensor = torch.nn.functional.fold(grad_wrt_weight_tensor, (weight.size()[2], weight.size()[3]), (1,1))
        print(grad_wrt_weight_tensor.size())
        sys.exit()

        # grad wrt input matrix 
        grad_wrt_input = correlate(grad_wrt_op, torch.flip(torch.flip(weight, [0]), [1]), padding, stride) 
        grad_wrt_input_tensor = from_dlpack(grad_wrt_input.toDlpack()).float()
        grad_wrt_input_tensor = torch.nn.functional.fold(grad_wrt_input_tensor, (ip.size()[2], ip.size()[3]), (1,1))

        # grad wrt bias 
        grad_wrt_bias_tensor = torch.sum(grad_wrt_op, (0,1,2), keepdim=True)
        # grad_wrt_bias_tensor = from_dlpack(grad_wrt_bias.toDlpack()).float()

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


