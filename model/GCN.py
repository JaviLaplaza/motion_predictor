#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        # self.act_f = nn.ReLU()


    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, input_n=50, output_n=25, kernel_n=10, num_stage=12, node_n=48, phase_pred=True, intention_pred=True):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.input_feature = input_feature
        self.phase_pred = phase_pred
        self.intention_pred = intention_pred

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        if phase_pred:
            input_feature += 1
            if output_n % 10 == 0:
                k = 4
            else:
                k = 3
            Lin = node_n
            Lout = output_n + kernel_n
            s = 1
            d = 1
            p = int(((Lout - 1) * s + 1 - Lin + d * (k - 1)) / 2)

            self.f_phase = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=hidden_feature, kernel_size=k, stride=s,padding=p, dilation=d),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_feature, out_channels=1, kernel_size=1, stride=s, padding=0, dilation=d),
                nn.Sigmoid()
            )
        if intention_pred:
            input_feature += 1
            if output_n % 10 == 0:
                k = 4
            else:
                k = 4
            Lin = node_n
            Lout = output_n + kernel_n
            s = 1
            d = 1
            p = int(((Lout - 1) * s + 1 - Lin + d * (k - 1)) / 2)

            self.f_intention = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=hidden_feature, kernel_size=k, stride=s, padding=p, dilation=d),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_feature, out_channels=5, kernel_size=1, stride=s, padding=0, dilation=d)
            )


        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        #self.act_f = nn.ReLU()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)

        phase = []
        if self.phase_pred:
            phase = torch.unsqueeze(y[:, :, -2], dim=1)
            phase = self.f_phase(phase)



        intention = []
        if self.intention_pred:
            intention = torch.unsqueeze(y[:, :, -1], dim=1)
            intention = self.f_intention(intention)
            intention = intention.permute((0, 2, 1))

        y = y[:, :, :self.input_feature] + x

        #print(f'phase dimensions: {phase.shape}')
        #print(f'intention dimensions: {intention.shape}')

        return y, phase, intention

class FusionGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(FusionGCN, self).__init__()

        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []

        #self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        #self.gc7 = GraphConvolution(hidden_feature, output_feature, node_n=node_n)
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.soft = nn.Softmax()

        self.fusion = torch.nn.Conv1d(in_channels=node_n, out_channels=node_n, kernel_size=(input_feature-20)+1)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        #print(f'x.shape as input: {x.shape}')
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        #y = self.gcbs[0](y)
        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        #print(f'y.shape before softmax: {y.shape}')
        #y = self.soft(y)
        #print(f'y.shape after softmax: {y.shape}')
        #print(f'y.shape before fusion: {y.shape}')
        y = self.fusion(y)
        #print(f'y.shape after fusion: {y.shape}')


        phase = 1
        intention = 1

        return y, phase, intention

if __name__== "__main__":
    batch, nodes, dct_n = 256, 27, 20*4
    #print(f'Input tensor dims: {batch, nodes, dct_n}')
    input = torch.rand((batch, nodes, dct_n))
    GCN = FusionGCN(input_feature=dct_n, hidden_feature=512, p_dropout=0.3, num_stage=2, node_n=nodes)
    prediction, phase, intention = GCN(input)
    #print(f'Prediction tensor dims: {prediction.shape}')
    print('Input tensor dims:' + batch, nodes, dct_n)
    input = torch.rand((batch, nodes, dct_n))
    GCN = FusionGCN(input_feature=dct_n, hidden_feature=512, p_dropout=0.3, num_stage=2, node_n=nodes)
    prediction, phase, intention = GCN(input)
    print('Prediction tensor dims:' + prediction.shape)
