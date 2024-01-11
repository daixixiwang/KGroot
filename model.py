import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, featureless=True,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., **kwargs):
        """

        :param input_dim: 输入维度，对应A[0]的行数，即图的结点数
        :param output_dim:  超参数，隐藏层的units数目
        :param support:  A 的长度 感觉应该是  边种类数 * 2 + 1
        :param featureless: 使用或者忽略输入的features
        :param init:
        :param activation: 激活函数
        :param weights:
        :param W_regularizer:
        :param num_bases:  使用的bases数量 （-1：all）
        :param b_regularizer:
        :param bias:
        :param dropout:
        :param kwargs:
        """
        super(GraphConvolution, self).__init__()
        # self.init = initializations.get(init)
        # self.activation = activations.get(activation)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = F.ReLU()
        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights
        self.featureless = featureless  # use/ignore input features
        self.dropout = dropout
        self.w_regularizer = nn.L1Loss()

        assert support >= 1

        # TODO

        self.bias = bias
        self.initial_weights = weights
        self.num_bases = num_bases

        # these will be defined during build()
        # self.input_dim = None
        if self.num_bases > 0:
            # 使用的bases数大于0
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.num_bases, self.output_dim, dtype=torch.float32, device=device))
            self.W_comp = nn.Parameter(torch.empty(self.support, self.num_bases, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.W_comp)  # 通过网络层时，输入和输出的方差相同
        else:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.support, self.output_dim, dtype=torch.float32, device=device))
        nn.init.xavier_uniform_(self.W)

        if self.bias:
            self.b = nn.Parameter(torch.empty(self.output_dim, dtype=torch.float32, device=device))
            nn.init.xavier_uniform_(self.b)
        """
        Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，
        这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），
        因为下次样本输入时它可能又得工作了
        """
        self.dropout = nn.Dropout(dropout)

    def get_output_shape_for(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.output_dim)
        return output_shape  # (batch_size, output_dim)

    def forward(self, inputs, mask=None):
        # inputs是 [x] + A； x是一个A[0]shape的稀疏矩阵，是一个特征矩阵
        features = torch.tensor(inputs[0], dtype=torch.float32, device=device)
        A = inputs[1:]  # list of basis functions
        #
        # 寻找矩阵所有不为0的值 sparse.find(a)[-1]
        # a.nonzero() 返回 每个非0元素 所在行和列
        A = [torch.sparse.FloatTensor(torch.LongTensor(a.nonzero())
                                      , torch.FloatTensor(sparse.find(a)[-1])
                                      , torch.Size(a.shape)).to(device)
             if len(sparse.find(a)[-1]) > 0 else torch.sparse.FloatTensor(a.shape[0], a.shape[1])
             for a in A]
        # convolve
        if not self.featureless:
            # 使用特征矩阵x  featureless = False
            supports = list()
            for i in range(self.support):
                # 稀疏矩阵相乘 最终维度是 【（结点数， 特征数），，，，】
                # 相当于 依据连接关系，将结点周围结点的特征相加最为其特征
                # logging.info("A[i]:{} feature:{}".format(A[i].shape, features.shape))
                supports.append(torch.spmm(A[i], features))
            # 按照维度1 拼接 即成为了一个大矩阵 (结点数， self.support * 特证数)  特证数是 feature(x)的列数,默认为结点数
            supports = torch.cat(supports, dim=1)
        else:
            # 不适用特征x featureless = True
            values = torch.cat([i._values() for i in A], dim=-1)
            temp_list = list()
            for i, j in enumerate(A):
                j_index = j._indices()
                j_index_0 = j._indices()[0]
                j_index_re = j._indices()[0].reshape(1, -1)

                tt = [j._indices()[0].reshape(1, -1),
                      (j._indices()[1] + (i * self.input_dim)).reshape(1, -1)]
                temp1 = torch.cat(tt)
                temp_list.append(temp1)
            indices = torch.cat(temp_list, dim=-1)
            # indices = torch.cat([torch.cat([j._indices()[0].reshape(1,-1),
            #                                (j._indices()[1] + (i*self.input_dim)).reshape(1,-1)])
            #              for i, j in enumerate(A)], dim=-1)
            print("featureless:{} indices:{} values:{}".format(self.featureless, indices.shape, values.shape))
            # 没有特征的输入，就将A[:]拼接 为 （结点数， A长度*结点数）的矩阵
            supports = torch.sparse.FloatTensor(indices, values, torch.Size([A[0].shape[0],
                                                                             len(A) * self.input_dim]))
        self.num_nodes = supports.shape[0]
        if self.num_bases > 0:

            V = torch.matmul(self.W_comp,
                             self.W.reshape(self.num_bases, self.input_dim, self.output_dim).permute(1, 0, 2))
            V = torch.reshape(V, (self.support * self.input_dim, self.output_dim))
            output = torch.spmm(supports, V)
        else:
            # （结点数， A长度* 特征数） * （输入维度*support, output_dim）
            output = torch.spmm(supports, self.W)

        # if featureless add dropout to output, by elementwise matmultiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = torch.ones(self.num_nodes, device=device)
            tmp_do = self.dropout(tmp)
            output = (output.transpose(1, 0) * tmp_do).transpose(1, 0)

        if self.bias:
            output += self.b
        return self.activation(output)


class GraphSimilarity(nn.Module):

    def __init__(self, input_dim, gcn_hidden_dim, linear_hidden_dim, num_bases, dropout, support=3, max_node_num=100):
        """
        计算两个图的相似度，即二分类，输出为(2, )
        :param input_dim: 初始每个结点的特征维度
        :param gcn_hidden_dim: gcn输出的每个结点的特征维度
        :param linear_hidden_dim: 多层感知机，中间隐层的神经单元数量
        :param num_bases: 默认为-1
        :param dropout: 舍弃dropout百分比的参数，不进行更新
        :param support: 采用多少个邻接矩阵，因为本项目中 因果事件图，只有3个邻接表
        :param max_node_num: 每个图的结点数目。不够的会进行padding
        """
        super(GraphSimilarity, self).__init__()

        # input_dim 需要与 feature列数一致
        self.gcn_online = GraphConvolution(input_dim, gcn_hidden_dim, num_bases=num_bases, activation="relu",
                                           featureless=False, support=support)
        self.gcn_kb = GraphConvolution(input_dim, gcn_hidden_dim, num_bases=num_bases, activation="relu",
                                       featureless=False, support=support)
        kernel_size = (max_node_num, 1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # 池化层输出的 向量 拼接后，为(gcn_hidden_dim*2, )
        self.linear_1 = torch.nn.Linear(in_features=gcn_hidden_dim*2, out_features=linear_hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=linear_hidden_dim, out_features=2, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_1, graph_2,  mask=None):
        # 两个图分别输入 RGCN 中，并最大池化
        gcn_info_1 = self.gcn_online(graph_1, mask=mask)
        gcn_info_1_drop = self.dropout(gcn_info_1)
        gcn_info_1_un = torch.unsqueeze(gcn_info_1_drop, 0)
        graph_info_1_pool = self.pool_2(gcn_info_1_un)
        graph_info_1_sq = torch.squeeze(graph_info_1_pool, 0)
        logging.info("gcn_info_1{} gcn_info_1_drop{} gcn_info_1_un{} graph_info_1_pool{} graph_info_1_sq{}".format(gcn_info_1.shape, gcn_info_1_drop.shape, gcn_info_1_un.shape,
                                             graph_info_1_pool.shape, graph_info_1_sq.shape))

        # graph_info_1 = self.pool_1(self.dropout(gcn_info_1))
        gcn_info_2 = self.gcn_kb(graph_2, mask=mask)
        gcn_info_2_drop = self.dropout(gcn_info_2)
        gcn_info_2_un = torch.unsqueeze(gcn_info_2_drop, 0)
        graph_info_2_pool = self.pool_2(gcn_info_2_un)
        graph_info_2_sq = torch.squeeze(graph_info_2_pool, 0)
        logging.info("gcn_info_2:{} gcn_info_2_drop:{} gcn_info_2_un:{} graph_info_2_pool:{} graph_info_2_sq:{}".format(
            gcn_info_2.shape, gcn_info_2_drop.shape, gcn_info_2_un.shape,
            graph_info_2_pool.shape, graph_info_2_sq.shape))
        # concat 输入多层感知机

        cat_info = torch.cat([graph_info_1_sq, graph_info_2_sq], dim=1)

        cat_info = self.linear_1(cat_info)
        cat_info = F.relu(cat_info)

        output = self.linear_2(cat_info)
        output = F.relu(output)
        # print(output.shape)
        output = torch.transpose(output, 0, 1)  # Shape: torch.Size([2, 71])

# Add batch dimension
        # output = output.unsqueeze(0)  # Shape: torch.Size([1, 2, 71])
        logging.info("output:{}".format(output.shape))
        # 输出维度为 （2，）tensor
        return output