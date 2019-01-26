import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super(Dense, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        return x


class CrossCompressUnit(nn.Module):
    def __init__(self, dim):
        super(CrossCompressUnit, self).__init__()
        self.dim = dim
        # 自定义参数
        # out_features在前，因为在F.linear中会先进行转置
        self.vv = nn.Parameter(torch.Tensor(1, dim))
        self.ev = nn.Parameter(torch.Tensor(1, dim))
        self.ve = nn.Parameter(torch.Tensor(1, dim))
        self.ee = nn.Parameter(torch.Tensor(1, dim))
        self.bias_v = nn.Parameter(torch.Tensor(dim))
        self.bias_e = nn.Parameter(torch.Tensor(dim))
        # 初始化正态分布
        # 参考:https://blog.csdn.net/qq_16234613/article/details/81604081
        stdv = 1. / math.sqrt(self.vv.size(1))
        self.vv.data.uniform_(-stdv, stdv)
        self.ev.data.uniform_(-stdv, stdv)
        self.ve.data.uniform_(-stdv, stdv)
        self.ee.data.uniform_(-stdv, stdv)
        self.bias_v.data.uniform_(-stdv, stdv)
        self.bias_e.data.uniform_(-stdv, stdv)

    def forward(self, inputs: list):
        # [batch_size, dim]
        v, e = inputs

        # unsqueeze的作用是在指定维度上增加一维
        # [batch_size, dim, 1], [batch_size, 1, dim]
        v = v.unsqueeze(2)
        e = e.unsqueeze(1)

        # [batch_size, dim, dim]
        c_matrix = v.matmul(e)
        c_matrix_transpose = c_matrix.permute([0, 2, 1]).contiguous()

        # [batch_size * dim, dim]
        c_matrix = c_matrix.view(-1, self.dim)
        c_matrix_transpose = c_matrix_transpose.view(-1, self.dim)

        # F.linear(input, weight, bias=None)
        # [batch_size, dim]
        v_out = F.linear(c_matrix, self.vv) + F.linear(c_matrix_transpose, self.ev)
        v_out = v_out.view(-1, self.dim) + self.bias_v

        e_out = F.linear(c_matrix, self.ve) + F.linear(c_matrix_transpose, self.ee)
        e_out = e_out.view(-1, self.dim) + self.bias_e

        return v_out, e_out
