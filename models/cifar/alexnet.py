'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import sys
import time
import math 
# from quantiser import quantise_inputs
from .quantisationblock import QuantisationBlock
from pyinn.modules import Conv2dDepthwise 
from customFuncs import Conv2d 


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        # self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv1 = Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv2 = Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        times = torch.FloatTensor([])
        
        # print("input size : ", x.size())
        # print("weight size : ", self.conv1.weight.size())
        # print("bias size : ", self.conv1.bias.size())
        # print("conv1")
        # print("input size : ", x.size())
        x = self.conv1(x)
        # print("output size : ", x.size())
        
        x = self.relu(x)
        
        x = self.maxpool2d(x)
        
        # print("input size : ", x.size())
        # print("weight size : ", self.conv2.weight.size())
        # print("bias size : ", self.conv2.bias.size())
        # tmp1 = torch.nn.functional.unfold(x, (5,5), padding=2)
        # tmp2 = self.conv2.weight.view(self.conv2.weight.size(0), -1)
        # print("input unfolded : ", tmp1.size()) 
        # print("weight unfolded : ", tmp2.size())
        # out1 = tmp1.transpose(1,2).matmul(tmp2.t())
        # print("output : ", out1.size()) 

        # tmp_bias = self.conv2.bias.repeat(out1.size()[0], out1.size()[1]).view(out1.size()[0], out1.size()[1], self.conv2.bias.size()[0])
        # print("bias repeated :", tmp_bias.size())
        # out1 += tmp_bias
        # out1 = out1.transpose(1,2)
        # print(out1.size())
        # 
        # h_in = x.size()[2]
        # w_in = x.size()[3] 
        # padding = (2,2)
        # dilation = (1,1) 
        # kernel = (5,5) 
        # stride = (1,1) 
        # o_dim_x = math.floor((((h_in + 2*padding[0]) - (dilation[0] * (kernel[0] - 1)) - 1)/(stride[0]))+1)
        # o_dim_y = math.floor((((w_in + 2*padding[1]) - (dilation[1] * (kernel[1] - 1)) - 1)/(stride[1]))+1)

        # out2 = torch.nn.functional.fold(out1, (o_dim_x, o_dim_y), (1,1))
        # print("folded output : ", out2.size())


        # sys.exit()
        
        x = self.conv2(x) 
        
        x = self.relu(x)
        
        x = self.maxpool2d(x)
        
        x = self.conv3(x) 
        
        x = self.relu(x)
        
        x = self.conv4(x) 
        
        x = self.relu(x)
        
        # print("conv5")
        # print("input size : ", x.size())
        x = self.conv5(x) 
        # print("output size : ", x.size())
        # print("weight size : ", self.conv5.weight.size())
        
        x = self.relu(x)
        
        x = self.maxpool2d(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
