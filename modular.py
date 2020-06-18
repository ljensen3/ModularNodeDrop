import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import reduce
import operator
import torch_sparse
import numpy as np

class ClampedRelu(nn.Module):
    def __init__(self, beta=10.0):
        super(ClampedRelu, self).__init__()
        self.sp = nn.Softplus(beta=beta)
        #self.shift = -1 + self.sp(1)
        #self.mult = 1./(1.+shift)
        #self.scaled_sp = sp 

    def forward(self, x):
        #return F.relu(self.mult*(1 - self.sp(1-x)+self.shift))
        return F.relu(1 - self.sp(1-x))


class NodeDrop(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True,reg_pam=0.0,batch_norm=True):
        super(NodeDrop, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reg_lambda=reg_pam
        self.batch_norm=batch_norm
        #self.reset_parameters()
        self.wlayer = nn.Linear(in_features,out_features)
        if self.batch_norm:
            self.blayer = nn.BatchNorm1d(out_features)
            self.alayer = nn.ReLU()
        else:
            self.blayer=nn.Identity(out_features)
            self.alayer =  ClampedRelu()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input2):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        #print(rett[0])
        ow=self.wlayer(input2)
        bw=self.blayer(ow)
        aw=self.alayer(bw)
        return aw
        
        
        return (activation + self.bias)*rett[0].float()
    """
    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)



    def train(self, mode):
        self.training = mode
        super(LinearARD, self).train(mode)
    """ 
    def get_reg(self, **kwargs):
        """
        #this version computes the sum of the norms
        zero = torch.zeros((), device=layer.weight.device)
        weights = torch.where(layer.weight<=0, zero, layer.weight).view(layer.weights.size(0), -1)
        weight_l1_loss = torch.sum(torch.norm(weights, dim=-1, p=power))
        bias_l1_loss = torch.sum(torch.where(layer.bias + self.args.reg_C),p=power)
        """
        power=1.0
        zero = torch.zeros((), device=self.wlayer.weight.device)
        self.weight,self.bias=self.wlayer.weight,self.wlayer.bias
        #this version computes the norm^pow eg l2^2
        if not self.batch_norm:
            weight_l1_loss = torch.sum(torch.where(self.weight<=0, zero, self.weight).pow(power))
            bpc = self.bias + 1.0e-1
            bias_l1_loss = torch.sum(torch.where(bpc<=0, zero, bpc).pow(power))

            reg_loss = self.reg_lambda * (weight_l1_loss + bias_l1_loss)
        else:
            varr='var'
            self.weight,self.bias=self.blayer.weight,self.blayer.bias
            if varr == 'var':
                weight_value = torch.abs(self.weight) * np.sqrt(self.weight.size(0))
            elif varr == 'l2':
                weight_value = torch.abs(self.weight)
            max_value = weight_value + self.bias
            vpc = max_value + 1.0e-1

            l1_loss = torch.sum(torch.where(vpc<=0, zero, vpc))

            reg_loss = self.reg_lambda * l1_loss

        return reg_loss



    def nodes(self):
        b = self.wlayer.bias
        w = self.wlayer.weight
        if isinstance(self.blayer, torch.nn.modules.batchnorm._BatchNorm):
            #have to handle the case for batch norms differently
            #abs(gamma) * sqrt(m) + b <= 0
            varr='var'
            b,w = self.blayer.bias,self.blayer.weight
            if varr == 'var':
                weight_sum = torch.abs(w) * np.sqrt(w.size(0))
                #print(torch.abs(w).size(),np.sqrt(w.size(0)))
            elif varr == 'l2':
                weight_sum = torch.abs(w)
        else:
            #sum(wx) + b <= 0 with 0 <= x <= 1
            w = w.view((w.size(0), -1))
            #print(w.size(),2)
            weight_sum=torch.sum(torch.max(w,torch.zeros((), device=w.device)), 1)
        #print(weight_sum.size(),b.size())
        value= weight_sum+b
        vect = (value>=0)
        count = torch.sum(vect)
        return count.item()

  
    """
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
    #    ""
    #    Get number of dropped weights (with log alpha greater than "thresh" parameter)
    #    :returns (number of dropped weights, number of all weight)
    #    ""
        return self.get_clip_mask().sum().cpu().numpy()


    def get_node_count(self):
        rett= self.out_features-self.get_clip_mask().sum()/self.in_features
        #print(rett)
        return rett

    def node_count(self):
        return self.in_features
    """





class NodeDropConv(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """


    def __init__(self, in_features, out_features, bias=True,reg_pam=0.0,batch_norm=True,kernel_size=3,stride=1,padding=(1,1)):
        super(NodeDropConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reg_lambda=reg_pam
        self.batch_norm=batch_norm
        #self.reset_parameters()
        self.wlayer = nn.Conv2d(in_features,out_features,kernel_size=kernel_size,stride=stride,padding=padding)
        if self.batch_norm:
            self.blayer = nn.BatchNorm2d(out_features)
            self.alayer = nn.ReLU()
        else:
            self.blayer=nn.Identity(out_features)
            self.alayer =  ClampedRelu()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input2):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        #print(rett[0])
        ow=self.wlayer(input2)
        bw=self.blayer(ow)
        aw=self.alayer(bw)
        return aw
        
        
        return (activation + self.bias)*rett[0].float()
    """
    def reset_parameters(self):
        self.weight.data.normal_(std=0.01)
        if self.bias is not None:
            self.bias.data.uniform_(0, 0)



    def train(self, mode):
        self.training = mode
        super(LinearARD, self).train(mode)
    """ 
    def get_reg(self, **kwargs):
        """
        #this version computes the sum of the norms
        zero = torch.zeros((), device=layer.weight.device)
        weights = torch.where(layer.weight<=0, zero, layer.weight).view(layer.weights.size(0), -1)
        weight_l1_loss = torch.sum(torch.norm(weights, dim=-1, p=power))
        bias_l1_loss = torch.sum(torch.where(layer.bias + self.args.reg_C),p=power)
        """
        power=1.0
        zero = torch.zeros((), device=self.wlayer.weight.device)
        self.weight,self.bias=self.wlayer.weight,self.wlayer.bias
        #this version computes the norm^pow eg l2^2
        if not self.batch_norm:
            weight_l1_loss = torch.sum(torch.where(self.weight<=0, zero, self.weight).pow(power))
            bpc = self.bias + 1.0e-1
            bias_l1_loss = torch.sum(torch.where(bpc<=0, zero, bpc).pow(power))

            reg_loss = self.reg_lambda * (weight_l1_loss + bias_l1_loss)
        else:
            varr='var'
            self.weight,self.bias=self.blayer.weight,self.blayer.bias
            if varr == 'var':
                weight_value = torch.abs(self.weight) * np.sqrt(self.weight.size(0))
            elif varr == 'l2':
                weight_value = torch.abs(self.weight)
            max_value = weight_value + self.bias
            vpc = max_value + 1.0e-1

            l1_loss = torch.sum(torch.where(vpc<=0, zero, vpc))

            reg_loss = self.reg_lambda * l1_loss

        return reg_loss

    def nodes(self):
        b = self.wlayer.bias
        w = self.wlayer.weight
        if isinstance(self.blayer, torch.nn.modules.batchnorm._BatchNorm):
            #have to handle the case for batch norms differently
            #abs(gamma) * sqrt(m) + b <= 0
            varr='var'
            b,w = self.blayer.bias,self.blayer.weight 
            if varr == 'var':
                weight_sum = torch.abs(w) * np.sqrt(w.size(0))
                #print(torch.abs(w).size(),np.sqrt(w.size(0)))
            elif varr == 'l2':
                weight_sum = torch.abs(w)
        else:
            #sum(wx) + b <= 0 with 0 <= x <= 1
            w = w.view((w.size(0), -1))
            #print(w.size(),2)
            weight_sum=torch.sum(torch.max(w,torch.zeros((), device=w.device)), 1)
        #print(weight_sum.size(),b.size())
        value= weight_sum+b
        vect = (value>=0)
        count = torch.sum(vect)
        return count.item()



def get_ard_reg(module, reg=0):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, NodeDrop) or isinstance(module, NodeDropConv): return reg + module.get_reg()
    if hasattr(module, 'children'): return reg + sum([get_ard_reg(submodule) for submodule in module.children()])
    return reg


