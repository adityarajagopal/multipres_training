import math 
import sys
import cupy as cp 
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module 
from torch.nn.parameter import Parameter 
from torch.utils.dlpack import to_dlpack, from_dlpack
from pyinn.utils import Dtype, Stream, load_kernel


CUDA_NUM_THREADS = 1024
matmul_kernel = '''
extern "C"
__global__ void gpu_matrix_mult(signed char *a, signed char *b, signed char *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    signed char sum = 0;

    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            // sum += a[row * n + i] * b[i * k + col];
            // edited to be able to deal with the transpose run needed for this to work
            sum += a[i * k + col] * b[row * n + i];
        }
        c[row * k + col] = sum;
    }
}
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

def cuda_correlate(ip, weight, padding, stride, kernel_size, corner_case=False, op_channels=None, dilation=1) : 
    if corner_case:
        w_transform = torch.flip(weight, torch.arange(weight.dim()).tolist())
    o_channels = weight.size(0)
    ip_unfold = torch.nn.functional.unfold(ip, (kernel_size, kernel_size), padding=padding, stride=stride)
    w_unfold = weight.view(o_channels, -1) 
    
    # setup output
    batch_size, channels, height, width = ip.size()

    kernel_h, kernel_w = weight.size()[2:]
    output_h = int((height + 2 * padding - (dilation * (kernel_h - 1) + 1)) / stride + 1)
    output_w = int((width + 2 * padding - (dilation * (kernel_w - 1) + 1)) / stride + 1)
    output = ip.new(batch_size, weight.size(0), output_h, output_w)
    op_unfold = output.view(ip.size(0), weight.size(0), output_h*output_w).byte()

    # cupy conversion 
    ip_cupy = cp.fromDlpack(to_dlpack(ip_unfold)).astype('int8')
    w_cupy = cp.fromDlpack(to_dlpack(w_unfold)).astype('int8')
    op_cupy = cp.fromDlpack(to_dlpack(op_unfold)).astype('int8')

    # need to pretend like these are transposed, so swapping the values for m, n and k
    # should be: m = ip(1), n = ip(2) k = w(1)
    m = ip_unfold.size(2)
    n = ip_unfold.size(1)
    k = w_unfold.size(0)

    # set these up properly to make this work
    blockSize = 16
    batchNum = ip_unfold.size(0)
    dimBlock = (blockSize, blockSize, 1)
    dimGrid = (int((k + blockSize - 1)/ blockSize), int((m + blockSize - 1)/ blockSize), 1)
    # print("batchNum: ", batchNum, "dimBlock:", dimBlock, "dimGrid:", dimGrid)

    f = load_kernel('gpu_matrix_mult', matmul_kernel)
    f(block=dimBlock, 
      grid=dimGrid,
      args=[ip_cupy.data.ptr, w_cupy.data.ptr, op_cupy.data.ptr, m, n, k],
      stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

    # multiplication in cupy 
    op = cp.matmul(cp.transpose(ip_cupy, (0,2,1)), cp.transpose(w_cupy))
    op = cp.transpose(op, (0,2,1))

    return op 
