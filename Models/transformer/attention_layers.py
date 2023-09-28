import copy
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.fft as fft
import paddle.nn.functional as F
from functools import partial
from Models.configs import *

def trans_torch(ndim, dim0, dim1):
    perm = list(range(ndim))
    temp = perm[dim0]
    perm[dim0] = perm[dim1]
    perm[dim1] = temp

    return perm

def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.shape[-1]

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
    else:
        scores = paddle.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        seq_len = scores.shape[-1]

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, axis=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = paddle.matmul(p_attn, value)

    return out, p_attn


def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.shape[-2]
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    perm = trans_torch(4,-2,-1)
    scores = paddle.matmul(key.transpose(perm), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = paddle.matmul(query, p_attn)
    return out, p_attn


def causal_linear_attn(query, key, value, kv_mask=None, dropout=None, eps=1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask

    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

    p_attn = paddle.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim=-3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / paddle.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = paddle.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn


class PositionalEncoding(nn.Layer):
    '''
    https://pypaddle.org/tutorials/beginner/transformer_tutorial.html
    This is not necessary if spacial coords are given
    input is (batch, seq_len, d_model)
    '''

    def __init__(self, d_model,
                 dropout=0.1,
                 max_len=2 ** 13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = paddle.zeros(max_len, d_model)
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = paddle.exp(paddle.arange(
            0, d_model, 2).float() * (-math.log(2 ** 13) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, min_freq=1 / 64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (paddle.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = paddle.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return paddle.concat((freqs, freqs), axis=-1)  # [b, n, d]


class FeedForward(nn.Layer):
    """
    FeedForward layer in transformers
    """

    def __init__(self, in_dim=256,
                 dim_feedforward: int = 1024,
                 out_dim=None,
                 batch_norm=False,
                 activation='relu',
                 dropout=0.1):
        super(FeedForward, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        n_hidden = dim_feedforward
        # activation = default(activation, 'relu')
        self.lr1 = nn.Linear(in_dim, n_hidden)
        self.activation = activation_dict[activation]
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1D(n_hidden)
        self.lr2 = nn.Linear(n_hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        forward compute
        :param in_var: (batch, seq_len, in_dim)
        """
        x = self.activation(self.lr1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = x.permute((0, 2, 1))
            x = self.bn(x)
            x = x.permute((0, 2, 1))
        x = self.lr2(x)
        return x


class SimpleAttention(nn.Layer):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pypaddle implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pypaddle/pypaddle/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = paddle.einsum('bhnd,bhne->bhde', k, v)
    attn = paddle.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.LayerList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key
        perm = trans_torch(4,1,2)
        query, key, value = \
            [layer(x).reshape([bsz, -1, self.n_head, self.d_k]).transpose(perm)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                value = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                query = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.shape[-1] == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.tile([1, self.n_head, 1, 1])
            query, key, value = [paddle.concat([pos, x], axis=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                     mask=mask,
                                                     dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        perm = trans_torch(4,1,2)
        att_output = x.transpose(perm).reshape([bsz, -1, out_dim])

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        """
        weight initialize
        """
        for param in self.linears.parameters():
            if param.ndim > 1:
                nn.initializer.XavierUniform(param)
                if self.diagonal_weight > 0.0:
                    param += self.diagonal_weight * \
                                  paddle.diag(paddle.ones(
                                      param.shape[-1], dtype='float32'))
                if self.symmetric_init:
                    param += param.T
                    # param.data /= 2.0
            else:
                nn.initializer.Constant(value=param)

    def _get_norm(self, eps):
        """
        batch/layer/instance normalization
        """
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        """
        layer normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        """
        instance normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


class CrossAttention(nn.Layer):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pypaddle implementation

    attn_types:
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pypaddle/pypaddle/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = paddle.einsum('bhnd,bhne->bhde', k, v)
    attn = paddle.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm_add=False,
                 norm_type='layer',
                 eps=1e-5):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.LayerList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        if self.xavier_init > 0:
            self._reset_parameters()
        self.norm_add = norm_add
        self.norm_type = norm_type
        if norm_add:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head * pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        """
        forward compute
        :param query: (batch, seq_len, d_model)
        :param key: (batch, seq_len, d_model)
        :param value: (batch, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.shape[0]
        if weight is not None:
            query, key = weight * query, weight * key

        query, key, value = \
            [layer(x).reshape([bsz, -1, self.n_head, self.d_k]).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.norm_add:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                value = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], axis=1)
                query = paddle.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], axis=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.shape[-1] == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [paddle.concat([pos, x], axis=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                     mask=mask,
                                                     dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
                                                             (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).reshape([bsz, -1, out_dim])

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output, self.attn_weight

    def _reset_parameters(self):
        """
        weight initialize
        """
        for param in self.linears.parameters():
            if param.ndim > 1:
                # xavier_uniform_(param, gain=self.xavier_init)
                nn.initializer.XavierUniform(param)
                if self.diagonal_weight > 0.0:
                    param += self.diagonal_weight * \
                                  paddle.diag(paddle.ones(
                                      param.shape[-1], dtype='float32'))
                if self.symmetric_init:
                    param += param.data.T
                    # param.data /= 2.0
            else:
                nn.initializer.Constant(param)

    def _get_norm(self, eps):
        """
        batch/layer/instance normalization
        """
        if self.attention_type in ['linear', 'galerkin', 'global']:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_V = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_V = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
        else:
            if self.norm_type == 'instance':
                self.norm_K = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
                self.norm_Q = self._get_instancenorm(self.d_k, self.n_head,
                                                     eps=eps,
                                                     affine=True)
            elif self.norm_type == 'layer':
                self.norm_K = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)
                self.norm_Q = self._get_layernorm(self.d_k, self.n_head,
                                                  eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        """
        layer normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        """
        instance normalization
        """
        return nn.LayerList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


if __name__ == '__main__':
    Q = paddle.ones([10, 100, 512])
    K = paddle.ones([10, 100, 512])
    V = paddle.ones([10, 100, 512])
    layer = SimpleAttention(n_head=8, d_model=512, norm_type='instance', norm_add=True)
    y = layer(Q, K, V)
    print(y)
