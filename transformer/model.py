from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F



"""reference: https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51
"""
def scaled_dot_product_attention(query, key, value):
    """
    query, key, value 的维度：(batche_size, seq_len, feature_dim)
    """
    temp = query.bmm(key.transpose(1,2))

    # 参数-1指最后一个维度
    scale = query.size(dim=-1) ** 0.5

    # 参数-1指最后一个维度
    softmax = F.softmax(temp/scale, dim=-1)

    return softmax.bmm(value)


class AttentionHead(nn.Module):
    """
    AttentionHead 相比于 scaled_dot_product_attention，
        增加输入层，即对 query, key, value 投影层，用全连接层表示
        输出就是 scaled_dot_product_attention 的输出
    """
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, q, k, v):
        query = self.q(q)
        key = self.k(k)
        value = self.v(v)
        return scaled_dot_product_attention(query, key, value)


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention 要将单个 AttentionHead 的输出拼接起来，
        先指定head数，其他输入和AttentionHead一样
    """
    def __init__(self, num_heads, dim_in, dim_q, dim_k):
        super().__init__()
        self.heads = nn.ModuleList(
            [ AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads) ]
        )
        """这里全连接层 的 输入维度，实质等于 num_heads*dim_v，
            因为 AttentionHead 最终是对 value 加权
            而 dim_k=dim_v，所以 num_heads*dim_k=num_heads*dim_v
        """
        self.linear = nn.Linear(num_heads*dim_k, dim_in)

    def forward(self, query, key, value):
        return self.linear(
            # 向量拼接，dim=-1表示在最后一个维度拼接，最后一个维度是 特征维度，形象地说就是 一个词用多少维向量表示
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


def position_encoding(seq_len, dim_feature, device=torch.device('cpu')):
    # 位置编码，其实直接可以设成 (batch_size, seq_len, dim_feature) 维度的向量，当作可学习参数
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_feature, dtype=torch.float, device=device).reshape(1, 1, -1)

    phase = pos / (10000**(dim // dim_feature))

    return torch.where(dim.long()%2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input, dim_hidden):
    """
    encoder/decoder 中的前馈神经网络，其实就是两个全连接层，
    第一个全连接层输入的维度，其实等于 MultiHeadAttention 输出的特征维度，
    中间隐层的维度是 dim_hidden，输出的维度等于第一个全连接层输入维度
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_hidden),
        nn.ReLU(),
        nn.Linear(dim_hidden, dim_input)
    )


class ResidualAndNorm(nn.Module):
    """
    encoder/decoder 中的残差连接和归一化操作
    """
    def __init__(self, sublayer, dimension, dropout):
        super().__init__()
        # sublayer 是干嘛的？
        self.sublayer = sublayer
        # 这里传的参和最新文档不一致
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        # 明白了似乎，tensors 有多个 nn.Module，第一个就是输入，用 tensors[0] 表示
        x = tensors[0]
        return self.norm(x + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_feature, num_heads, dim_hidden, dropout):
        super().__init__()

        dim_q = dim_k = max(dim_feature//num_heads, 1)

        self.attention = ResidualAndNorm(
            # 这里要理解一下，和ResidualAndNorm中的sublayer对应起来
            MultiHeadAttention(num_heads, dim_feature, dim_q, dim_k),
            dimension=dim_feature,
            dropout=dropout
        )

        self.feed_forward = ResidualAndNorm(
            feed_forward(dim_feature, dim_hidden),
            dimension=dim_feature,
            dropout=dropout
        )

    def forward(self, src):
        # attention 输入是一样的，里面做了不同线性投影
        src = self.attention(src, src, src)
        return self.feed_forward(src)

        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, dim_feature=512, num_heads=8, dim_hidden=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [ TransformerEncoderLayer(dim_feature, num_heads, dim_hidden, dropout) for _ in range(num_layers) ]
        )
    
    def forward(self, src):
        seq_len, dimension = src.size(1), src.size(2)

        src += position_encoding(seq_len, dimension)

        for layer in self.layers:
            # 这个设计很妙，前一层的输出就是下一层的输入
            src = layer(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_feature=512, num_heads=6, dim_hidden=2048, dropout=0.1):
        super().__init__()

        dim_q = dim_k = max(dim_feature//num_heads, 1)
        # 参考的博客设置了两个完全相同的 self.attention，没必要
        self.attention = ResidualAndNorm(
            MultiHeadAttention(num_heads, dim_feature, dim_q, dim_k),
            dimension=dim_feature,
            dropout=dropout
        )

        self.feed_forward = ResidualAndNorm(
            feed_forward(dim_feature, dim_hidden),
            dimension=dim_feature,
            dropout=dropout
        )

    def forward(self, tgt, memory):
        # decoder中，每个layer有两次attention，一次feed forward，模型结构就是这样的

        # 第一个attention的输入全部来自 decoder
        tgt = self.attention(tgt, tgt, tgt)
        # 第二个attention的输入来自 encoder + decoder，encoder提供key value，decoder提供 query
        tgt = self.attention(tgt, memory, memory)
        return self.feed_forward(tgt)

    
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, dim_feature=512, num_heads=8, dim_hidden=2048, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList(
            [ TransformerDecoderLayer(dim_feature, num_heads, dim_hidden, dropout) for _ in range(num_layers) ]
        )

        self.linear = nn.Linear(dim_feature, dim_feature)

    def forward(self, tgt, memory):
        seq_len, dimension = tgt.size(1), tgt.size(2)

        tgt += position_encoding(seq_len, dimension)

        for layer in self.layers:
            # 这个设计很妙，前一层的输出就是下一层的输入
            tgt = layer(tgt, memory)

        # 最终输出前，还有个线性投影层
        return F.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, dim_feature=512, num_heads=6, dim_hidden=2048, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_feature=dim_feature,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
            dropout=dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_feature=dim_feature,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
            dropout=dropout
        )

    def forward(self, src, tgt):
        # 这个代码表明，先要把 self.encoder 走完，输出才会接入 decoder，而且还是接入decoder的每一层
        memory = self.encoder(src)
        return self.decoder(tgt, memory)
