import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):  ##继承整数枚举
    batch = 0
    seq = 1
    feature = 2


class AKTNet(nn.Module):    ###加括号的意思是继承括号中的类，这里继承神经网络模型
    def __init__(self, n_question, d_model, n_blocks, seqlen, kq_same, dropout, final_fc_dim=512, n_heads=8,
                 d_ff=2048, separate_qa=False):
        super(AKTNet, self).__init__()
        """
        Input:
            d_model: dimension of attention block   ##注意力块的维度，，即一个单词用多少维的向量表示
            final_fc_dim: dimension of final fully connected net before prediction   ##最终全连接层的维度
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully connected net inside the basic block  ##基本块内全连接网络的尺寸
        """
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.seqlen = seqlen
        embed_l = d_model
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)   ###问题嵌入
        self.positon_embed = nn.Embedding(self.seqlen + 1, embed_l)    ###嵌入位置信息
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)    ###回答的嵌入,,,,后者表示嵌入的维度，前者是变量个数
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks, d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same)  ##//向小取整：5//2=2

        self.out = nn.Sequential(               ###全连接层，，，，讲里面的运算按顺序结合起来输出
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

    def forward(self, q_data, qa_data, target, graph):
        # Batch First
        #graph_Linear = nn.Linear(2, self.seqlen)
        #graph = graph_Linear(graph)
        graph = graph.T
        print('max',graph.max())
        print('min',graph.min())
        graph_embed = nn.Embedding(100 * self.seqlen + 1, 256)
        graph = graph_embed(graph).to(device)
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data - q_data) // self.n_question
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data
        l = list(range(self.seqlen))
        l = torch.tensor(l).to(device)
        L = self.positon_embed(l)
        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.model(q_embed_data, qa_embed_data, L, graph)   ###transformer的输出，，，，考虑在这里加上知识图
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = output.reshape(-1)
        mask = labels > -0.9
        masked_lables = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')   ###需不需要用新的损失函数迭代？
        output = loss(masked_preds, masked_lables)
        return output.sum(), m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(Architecture, self).__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks)    ####密集块数量
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data, position, graph):
        x = q_embed_data
        y = qa_embed_data
        p = position
        x = x + p
        y = y + p
        g = graph
        # encoder1
        for block in self.blocks_1:  # encode qas    编码器
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:   # decoder  解码器
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)  ###第一个密集块，不需要激活层
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)  ###第二个密集块，需要激活层
                flag_first = True

        # encoder2
        flag_first = True
        for block in self.blocks_2:   # decoder  解码器
            if flag_first:  # peek current question
                x = block(mask=1, query=x + g, key=x + g, values=x + g, apply_pos=False)  ###第一个密集块，不需要激活层
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x + g, key=x + g, values=x, apply_pos=True)  ###第二个密集块，需要激活层
                flag_first = True
        return x


class TransformerLayer(nn.Module):        ####Transformer层
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super(TransformerLayer, self).__init__()
        """
        This is a Basic Block of Transformer paper. It contains one Multi-head attention object. Followed by layer
        norm and position wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module).
                    It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            values: In transformer paper,
                    it is the input for encoder and encoded output for decoder (in masked attention part)
        Output:
            query: Input gets changed over the layer and returned.
        """
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')   ##取上三角矩阵，K表示是否需要移动对角线，，后两个表示矩阵尺寸，前面代表矩阵数量
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))   ##将结果先线性层、激活层等
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):    ###多头注意力机制
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttention, self).__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, self.gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):   ####单次的注意力机制
    """
    This is called by Multi-head attention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)    ###torch.matmul：矩阵乘法
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)   ###三维张量的尺寸？？

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()  ###返回一个内存张量，，，其中不改变X和Y轴的位置，即不变化，，，在这里即复制X1

    with torch.no_grad():   ###在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。

        ##在pytorch中，tensor有一个requires_grad参数，如果设置为True，则反向传播时，该tensor就会自动求导。
        # tensor的requires_grad的属性默认为False,若一个节点（叶子变量：自己创建的tensor）requires_grad被设置为True，
        # 那么所有依赖它的节点requires_grad都为True（即使其他相依赖的tensor的requires_grad = False）
        scores_ = scores.masked_fill(mask == 0, -1e32)   ##把等于0的用-1e32掩盖
        scores_ = F.softmax(scores_, dim=-1)    ###应用非线性激活函数
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)   ###进行累加操作
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)   ##返回所有元素之和
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill(mask == 0, -1e23)  ##进行掩码操作
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)   ###判断是否拼接个0张量，，，scores[:, :, 1:, :]从第一行开始算，去掉第0行
    scores = dropout(scores)   ###做dropout操作
    output = torch.matmul(scores, v)    ###公式中的最后一步
    return output