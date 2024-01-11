import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, featureless=True,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, num_bases=-1,
                 b_regularizer=None, bias=False, dropout=0., max_node_num=100, **kwargs)   :
        """

        :param input_dim: 输入维度，有特征矩阵时对应初始特征矩阵的词嵌入维度，无特征矩阵时，对应输入结点个数
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
            # nn.init.xavier_uniform_(self.W_comp)  # 通过网络层时，输入和输出的方差相同
            torch.nn.init.kaiming_normal_(self.W_comp, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            self.W = nn.Parameter(
                torch.empty(self.input_dim * self.support, self.output_dim, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.W)
        torch.nn.init.kaiming_normal_(self.W, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if self.bias:
            self.b = nn.Parameter(torch.empty(max_node_num, self.output_dim, dtype=torch.float32, device=device))
            # nn.init.xavier_uniform_(self.b)
            torch.nn.init.kaiming_normal_(self.b, a=0, mode='fan_in', nonlinearity='leaky_relu')
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
        """
        输入多张图
        :param inputs: ()
        :param mask:
        :return:
        """
        features, A_list = inputs[0], inputs[1]
        batch_size = features.shape[0]
        node_num = features.shape[1]
        feature_dim = features.shape[2]
        # 相当于 cir 进行归一化
        A_list = F.normalize(A_list, p=1, dim=3)
        if not self.featureless:
            A_list = A_list.narrow(dim=1, start=0, length=self.support)
            features = features.unsqueeze(dim=1)
            supports = torch.matmul(A_list, features)
            supports = supports.transpose(1, 2)
            supports = supports.reshape([batch_size, 1, node_num, self.support*feature_dim])
        else:
            supports = A_list.transpose(1, 2)
            supports = supports.reshape([batch_size, 1, node_num,  A_list.shape[1] * node_num])
        supports = supports.squeeze(dim=1)

        if self.num_bases > 0:

            V = torch.matmul(self.W_comp,
                             self.W.reshape(self.num_bases, self.input_dim, self.output_dim).permute(1, 0, 2))
            V = torch.reshape(V, (self.support * self.input_dim, self.output_dim))
            output = torch.matmul(supports, V)
        else:
            # （结点数， A长度* 特征数） * （输入维度*support, output_dim）
            output = torch.matmul(supports, self.W)

        # if featureless add dropout to output, by elementwise matmultiplying with column vector of ones,
        # with dropout applied to the vector of ones.
        if self.featureless:
            tmp = torch.ones(self.num_nodes, device=device)
            tmp_do = self.dropout(tmp)
            tmp_do_stack = torch.stack([tmp_do for i in range(output.shape[0])], dim=0)
            output = (output.transpose(1, 0) * tmp_do_stack).transpose(1, 0)

        if self.bias:
            output += self.b
        return output




class GraphSimilarity(nn.Module):

    def __init__(self, input_dim, gcn_hidden_dim, linear_hidden_dim, pool_step, num_bases, dropout, support=3, max_node_num=100):
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
                                           featureless=False, support=support, bias=True, max_node_num=max_node_num)
        self.gcn_kb = GraphConvolution(input_dim, gcn_hidden_dim, num_bases=num_bases, activation="relu",
                                       featureless=False, support=support, bias=True, max_node_num=max_node_num)
        pool_step = pool_step
        kernel_size = (max_node_num//pool_step, 1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size, stride=(max_node_num//pool_step, 1), padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size, stride=(max_node_num//pool_step, 1), padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # 池化层输出的 向量 拼接后，为(gcn_hidden_dim*2, )
        self.linear_1 = torch.nn.Linear(in_features=gcn_hidden_dim, out_features=linear_hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=linear_hidden_dim, out_features=2, bias=True)

        self.door = torch.nn.Parameter(torch.empty(1, pool_step, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.door)  # 通过网络层时，输入和输出的方差相同
        torch.nn.init.kaiming_normal_(self.door, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.node_choose_w = torch.nn.Parameter(
            torch.empty(1, pool_step, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.node_choose_w)  # 通过网络层时，输入和输出的方差相同
        torch.nn.init.kaiming_normal_(self.node_choose_w, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, graphs_1, graphs_2, mask=None):
        # 两个图分别输入 RGCN 中，并最大池化
        gcn_info_1 = self.gcn_online(graphs_1, mask=mask)
        gcn_info_1_drop = self.dropout(gcn_info_1)
        # gcn_info_1_drop = gcn_info_1
        gcn_info_1_ac = self.activation(gcn_info_1_drop)
        graph_info_1_pool = self.pool_1(gcn_info_1_ac)
        # logging.info("gcn_info_1{} gcn_info_1_drop{}  graph_info_1_pool{} ".format(
        #     np.array(gcn_info_1.shape), np.array(gcn_info_1_drop.shape), np.array(graph_info_1_pool.shape)))

        gcn_info_2 = self.gcn_kb(graphs_2, mask=mask)
        gcn_info_2_drop = self.dropout(gcn_info_2)
        # gcn_info_2_drop = gcn_info_2
        gcn_info_2_ac = self.activation(gcn_info_2_drop)
        graph_info_2_pool = self.pool_2(gcn_info_2_ac)
        # logging.info("gcn_info_2:{} gcn_info_2_drop:{} graph_info_2_pool:{} ".format(
        #     np.array(gcn_info_2.shape), np.array(gcn_info_2_drop.shape), np.array(graph_info_2_pool.shape)))

        # concat 输入多层感知机
        # 方式1： 前后相连
        # cat_info = torch.cat([graph_info_1_pool, graph_info_2_pool], dim=2)
        # 方式2： 门控制
        cat_info = (self.door * graph_info_1_pool.transpose(1, 2) + (1.0-self.door) * graph_info_2_pool.transpose(1, 2)
                    ).transpose(1, 2)

        # 方式3：zhuan两维以上向量
        # pass

        cat_info = torch.matmul(self.node_choose_w, cat_info).squeeze(dim=1)
        # cat_info = cat_info.squeeze(dim=1)

        cat_info = self.linear_1(cat_info)
        cat_info = self.activation(cat_info)

        output = self.linear_2(cat_info)
        # output = torch.nn.functional.softmax(output, dim=1)
        output = self.activation(output)
        # logging.info("output:{}".format(output.shape))
        # 输出维度为 （2，）tensor
        # assert int(output.shape[0]) == int(graphs_1[0].shape[0]) == int(graphs_2[0].shape[0])
        return output


class GraphSimilarity_No_Gcn(nn.Module):
    """去除gcn"""
    def __init__(self, input_dim, gcn_hidden_dim, linear_hidden_dim, pool_step, num_bases, dropout, support=3, max_node_num=100):
        super(GraphSimilarity_No_Gcn, self).__init__()

        # input_dim 需要与 feature列数一致

        pool_step = pool_step
        kernel_size = (max_node_num // pool_step, 1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size, stride=(max_node_num // pool_step, 1), padding=0, dilation=1,
                                         return_indices=False, ceil_mode=False)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size, stride=(max_node_num // pool_step, 1), padding=0, dilation=1,
                                         return_indices=False, ceil_mode=False)

        # 池化层输出的 向量 拼接后，为(gcn_hidden_dim*2, )
        self.linear_1 = torch.nn.Linear(in_features=input_dim, out_features=linear_hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=linear_hidden_dim, out_features=2, bias=True)

        self.door = torch.nn.Parameter(torch.empty(1, pool_step, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.door)  # 通过网络层时，输入和输出的方差相同
        torch.nn.init.kaiming_normal_(self.door, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.node_choose_w = torch.nn.Parameter(
            torch.empty(1, pool_step, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.node_choose_w)  # 通过网络层时，输入和输出的方差相同
        torch.nn.init.kaiming_normal_(self.node_choose_w, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        pass

    def forward(self,  graphs_1, graphs_2, mask=None):
        # 两个图分别输入 RGCN 中，并最大池化
        gcn_info_1 = graphs_1[0]
        gcn_info_2 = graphs_2[0]

        graph_info_1_pool = self.pool_1(gcn_info_1)
        graph_info_2_pool = self.pool_2(gcn_info_2)

        cat_info = (self.door * graph_info_1_pool.transpose(1, 2) + (1.0 - self.door) * graph_info_2_pool.transpose(1,
                                                                                                                    2)
                    ).transpose(1, 2)

        cat_info = torch.matmul(self.node_choose_w, cat_info).squeeze(dim=1)
        # cat_info = cat_info.squeeze(dim=1)

        cat_info = self.linear_1(cat_info)
        cat_info = self.activation(cat_info)

        output = self.linear_2(cat_info)
        # output = torch.nn.functional.softmax(output, dim=1)
        output = self.activation(output)
        # logging.info("output:{}".format(output.shape))
        # 输出维度为 （2，）tensor
        # assert int(output.shape[0]) == int(graphs_1[0].shape[0]) == int(graphs_2[0].shape[0])
        return output
        pass


class GraphSimilarity_No_KB(nn.Module):
    """去除KB 直接输入图之后 进行分类"""
    def __init__(self, input_dim, gcn_hidden_dim, linear_hidden_dim, out_dim, pool_step, num_bases, dropout, support=3, max_node_num=100):
        super(GraphSimilarity_No_KB, self).__init__()

        # input_dim 需要与 feature列数一致
        self.gcn_online = GraphConvolution(input_dim, gcn_hidden_dim, num_bases=num_bases, activation="relu",
                                           featureless=False, support=support, bias=True, max_node_num=max_node_num)

        pool_step = pool_step
        kernel_size = (max_node_num//pool_step, 1)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size, stride=(max_node_num//pool_step, 1), padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # 池化层输出的 向量 拼接后，为(gcn_hidden_dim*2, )
        self.linear_1 = torch.nn.Linear(in_features=gcn_hidden_dim, out_features=linear_hidden_dim, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=linear_hidden_dim, out_features=out_dim, bias=True)

        self.node_choose_w = torch.nn.Parameter(
            torch.empty(1, pool_step, dtype=torch.float32, device=device))
        # nn.init.xavier_uniform_(self.node_choose_w)  # 通过网络层时，输入和输出的方差相同
        torch.nn.init.kaiming_normal_(self.node_choose_w, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu


    def forward(self,  graphs_1, graphs_2, mask=None):
        gcn_info_1 = self.gcn_online(graphs_1, mask=mask)
        gcn_info_1_drop = self.dropout(gcn_info_1)
        gcn_info_1_ac = self.activation(gcn_info_1_drop)
        graph_info_1_pool = self.pool_1(gcn_info_1_ac)

        cat_info = torch.matmul(self.node_choose_w, graph_info_1_pool).squeeze(dim=1)

        cat_info = self.linear_1(cat_info)
        cat_info = self.activation(cat_info)

        output = self.linear_2(cat_info)
        # output = torch.nn.functional.softmax(output, dim=1)
        output = self.activation(output)
        return output


if __name__ == '__main__':
    a = [
        [[[1,1],[2,2]],
         [[3,3],[4,4]],
         [[5,5],[6,6]]],

        [[[7,7],[8,8]],
         [[9,9],[9,9]],
         [[10,10],[10,10]]
         ]
    ]
    b = [
        [[1,1,1],
         [2,2,2]],
        [[3,3,3],
          [4,4,4]]
    ]

    c = [[1,1],
         [2,2],
         [3,3]]




    tensor_a = torch.as_tensor(a, dtype=torch.float32)
    # logging.info("narrow:{} after:\n{}".format(tensor_a, tensor_a.narrow( dim=1, start=0, length=1)))
    tensor_b = torch.as_tensor(b, dtype=torch.float32)
    tensor_c = torch.as_tensor(c, dtype=torch.float32)
    # tensor_b = tensor_b.unsqueeze(dim=1)
    tensor_mal = torch.matmul(tensor_b, tensor_c)
    print(tensor_a.shape)
    print(tensor_b.shape)
    print(tensor_c.shape)
    print(tensor_mal.shape)
    # logging.info("tensor_a:{} ".format(tensor_a))
    print(tensor_b)
    print(tensor_c)
    print("tensor_mal:{}".format(tensor_mal))

    tmp = torch.ones(2)
    drop = nn.Dropout(0.5)
    tmp_do = drop(tmp)
    print("tmp_do:{}".format(tmp_do))
    tmp_do_stack = torch.stack([tmp_do for i in range(tensor_mal.shape[0])], dim=0)
    output1 = (tensor_mal[0].transpose(1, 0) * tmp_do).transpose(1, 0)
    tensor_mal_trans = tensor_mal.transpose(1,2)
    output2 = (tensor_mal_trans * tmp_do_stack).transpose(1, 2)
    print("output1:{}".format(output1))
    print("output2:{}".format(output2))

    bias = torch.as_tensor([1,2,3], dtype=torch.float32)
    print("b:{}".format(tensor_b))
    print("bias:{}".format(bias))
    print("sum:{}".format(tensor_b + bias))

    b_2 = [
        [[1,1,1],
         [2,2,2],
         [1,1,1],
         [2,2,2]],
        [[3,3,3],
          [4,4,4],
         [3,3,3],
         [4,4,4]]
    ]
    tensor_b_2 = torch.as_tensor(b_2, device=device, dtype=torch.float32)
    torch.rand((4,), dtype=torch.float32, device=device)
    tensor_b_2.mul()

    pool = torch.nn.MaxPool2d((2,1), stride=(2,1), padding=0, dilation=1, return_indices=False,
                       ceil_mode=False)
    pool_r = pool.forward(tensor_b_2)
    print("tensor_b_2:{}".format(tensor_b_2))
    print("pool_r:{}".format(pool_r))
    rr = torch.cat([pool_r, pool_r], dim=2)
    cat = rr.squeeze(dim=1)
    print("cat:{}".format(cat))
    linear = torch.nn.Linear(in_features= 6, out_features=2, bias=True)
    pre = linear.forward(cat)
    print("pre:{}".format(pre))

    linear2 = torch.nn.Linear(in_features=3, out_features=2, bias=True)
    batch_2 = linear2.forward(tensor_b)
    print("batch_2:{}".format(batch_2))

    # y = tensor_mal.transpose(1,2)
    # print(y)
    # y = y.reshape([2, 1, 2, 9])
    # print(y)

    # 归一化
    m = [
        [[[0,1],[1,1]],
         [[2,3],[2,4]],
         [[5,5],[4,6]]],

        [[[7,7],[0,8]],
         [[9,3],[9,9]],
         [[1,10],[10,10]]
         ]
    ]
    tensor_m = torch.as_tensor(m, dtype=torch.float32)
    m_nor = F.normalize(tensor_m, p=1, dim=3)
    print("normal:{}".format(m_nor))