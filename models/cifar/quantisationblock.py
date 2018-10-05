'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''
import torch.nn as nn
from quantiser import quantise_inputs 

class QuantisationBlock (nn.Module) : 
    def __init__ (self) : 
        super(QuantisationBlock, self).__init__()

    def forward (self, x) : 
        return quantise_inputs(x)
