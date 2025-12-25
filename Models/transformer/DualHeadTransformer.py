#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Time    : 2024/12/24
# @File    : DualHeadTransformer.py
# @Description: 双头条件化 Transformer
#   - G (条件场，不可叠加): 包含物理尺度/pos 信息，用于调制 U 的表征
#   - U (源项场，可叠加): 待编码的场变量
#   - 通过 CondLayerNorm (FiLM) 在每层 encoder 中用 G 调制 U
#   - 不使用 cross-attention，不将 G/U concat

使用示例:
----------
1. 基本使用:
    from transformer.DualHeadTransformer import DualHeadFourierTransformer
    
    config = dict(
        G_dim=2,           # G 的通道数 (含 pos 信息，如 x, y 坐标)
        U_dim=3,           # U 的通道数 (源项场变量)
        n_targets=1,       # 输出通道数 (如温度场)
        n_hidden=96,       # token 隐藏维度
        num_encoder_layers=4,
        n_head=4,
        attention_type='fourier',
        decoder_type='pointwise',
        spacial_dim=2,     # 2D or 3D
    )
    
    model = DualHeadFourierTransformer(**config)
    
    # 前向传播 (channel-last 格式)
    G = torch.randn(batch, H, W, 2)  # 条件场: [B, H, W, C]
    U = torch.randn(batch, H, W, 3)  # 源项场: [B, H, W, C]
    T = model(G, U)                  # 输出: [batch, H, W, 1]

2. 从 FourierTransformer 迁移:
    如果你之前使用 FourierTransformer 的配置 (node_feats=5)，
    需要拆分为 G_dim 和 U_dim:
    
    # 旧配置 (FourierTransformer)
    old_config = dict(
        node_feats=5,      # G(2通道) + U(3通道) 拼接
        n_targets=1,
        ...
    )
    
    # 新配置 (DualHeadFourierTransformer)
    new_config = dict(
        G_dim=2,           # G 的通道数
        U_dim=3,           # U 的通道数
        n_targets=1,
        ...
    )
    
    # 输入也需要拆分，且使用 channel-last 格式
    # 旧: node = torch.cat([G, U], dim=-1)  # [B, H, W, 5]
    # 新: model(G, U)                       # G:[B,H,W,2], U:[B,H,W,3]

3. 配置参数说明:
    必需参数:
        - G_dim: G 的通道数 (条件场，包含 pos 信息)
        - U_dim: U 的通道数 (源项场)
        - n_targets: 输出通道数
    
    可选参数:
        - n_hidden: token 隐藏维度 (默认 96)
        - num_encoder_layers: encoder 层数 (默认 4)
        - n_head: 注意力头数 (默认 4)
        - attention_type: 'fourier', 'galerkin', 'linear', 'softmax' (默认 'fourier')
        - decoder_type: 'pointwise', 'ifft' (默认 'pointwise')
        - spacial_dim: 2 or 3 (默认 2)
        - dropout: dropout 比例 (默认 0.05)
        - debug: 是否打印调试信息 (默认 False)
"""

import copy
import torch
import torch.nn as nn
from torch.nn.init import zeros_

from collections import defaultdict

from configs import default
from transformer.attention_layers import SimpleAttention, FeedForward
from transformer.Transformers import PointwiseRegressor, SpectralRegressor


class CondLayerNorm(nn.Module):
    """
    条件化 LayerNorm: 用条件向量 cond 调制归一化后的特征
    
    机制:
        - 先对 x 做标准 LayerNorm (无可学习参数)
        - 用 cond 预测 gamma 和 beta: gamma = 1 + gamma_pred, beta = beta_pred
        - 输出: gamma * LayerNorm(x) + beta
    
    初始化:
        - 条件映射层的权重和偏置初始化为 0
        - 保证训练初期 gamma=1, beta=0，等价于标准 LayerNorm
    
    Args:
        normalized_shape: 归一化的特征维度 (通常是 d_model)
        cond_dim: 条件向量的维度
        eps: LayerNorm 的数值稳定项
    
    Input:
        x: [B, N, C] - 待归一化的特征
        cond: [B, N, Cc] 或 [B, Cc] - 条件向量
    
    Output:
        [B, N, C] - 调制后的特征
    """
    
    def __init__(self, normalized_shape, cond_dim, eps=1e-5):
        super(CondLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.cond_dim = cond_dim
        self.eps = eps
        
        # LayerNorm (不带可学习参数)
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)
        
        # 条件映射: cond_dim -> 2*normalized_shape (gamma 和 beta)
        self.cond_proj = nn.Linear(cond_dim, 2 * normalized_shape)
        
        # 初始化为 0: 使得初始 gamma=1, beta=0
        zeros_(self.cond_proj.weight)
        zeros_(self.cond_proj.bias)
    
    def forward(self, x, cond):
        """
        Args:
            x: [B, N, C] - 输入特征
            cond: [B, N, Cc] 或 [B, Cc] - 条件向量
        
        Returns:
            [B, N, C] - 调制后的特征
        """
        # 标准归一化
        x_norm = self.norm(x)  # [B, N, C]
        
        # 预测 gamma 和 beta
        gamma_beta = self.cond_proj(cond)  # [B, N, 2C] or [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # 各 [B, N, C] or [B, C]
        
        # gamma 偏移使初始为 1
        gamma = 1.0 + gamma
        
        # 如果 cond 是 [B, Cc]，需要 broadcast 到 [B, N, C]
        if cond.dim() == 2:  # [B, Cc]
            gamma = gamma.unsqueeze(1)  # [B, 1, C]
            beta = beta.unsqueeze(1)    # [B, 1, C]
        
        # 调制
        return gamma * x_norm + beta


class ConditionalTransformerEncoderLayer(nn.Module):
    """
    条件化 Transformer Encoder Layer
    
    结构 (PostNorm 风格):
        x = CondLN1(x + Dropout(Attn(x)), cond)
        x = CondLN2(x + Dropout(FF(x)), cond)
    
    与 SimpleTransformerEncoderLayer 的区别:
        - 将 LayerNorm 替换为 CondLayerNorm
        - 每层用 cond (来自 g_tok) 调制 x (来自 u_tok)
    
    Args:
        d_model: token 的隐藏维度
        cond_dim: 条件向量的维度 (通常等于 d_model)
        n_head: 注意力头数
        dim_feedforward: FeedForward 的中间维度
        attention_type: 注意力类型 ('fourier', 'galerkin', 'linear', 'softmax')
        pos_dim: 位置编码维度 (如果需要)
        dropout: dropout 比例
        ffn_dropout: FeedForward 的 dropout
        activation_type: 激活函数类型
        debug: 是否打印调试信息
    
    Input:
        x: [B, N, d_model] - 输入 token (u_tok)
        cond: [B, N, cond_dim] - 条件 token (g_tok)
        pos: [B, N, pos_dim] - 位置信息 (可选)
        weight: [B, N, N] 或 [B, N] - 质量矩阵 (可选)
    
    Output:
        [B, N, d_model] - 输出 token
    """
    
    def __init__(self,
                 d_model=96,
                 cond_dim=96,
                 pos_dim=2,
                 n_head=4,
                 dim_feedforward=384,
                 attention_type='fourier',
                 xavier_init=1e-2,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 activation_type='silu',
                 dropout=0.05,
                 ffn_dropout=None,
                 norm_eps=1e-5,
                 debug=False):
        super(ConditionalTransformerEncoderLayer, self).__init__()
        
        dropout = default(dropout, 0.05)
        ffn_dropout = default(ffn_dropout, dropout)
        dim_feedforward = default(dim_feedforward, 2 * d_model)
        
        # 注意力层 (复用现有实现)
        self.attn = SimpleAttention(
            n_head=n_head,
            d_model=d_model,
            attention_type=attention_type,
            diagonal_weight=diagonal_weight,
            xavier_init=xavier_init,
            symmetric_init=symmetric_init,
            pos_dim=pos_dim,
            norm_add=False,  # 我们用 CondLN 替代
            dropout=dropout
        )
        
        # FeedForward 层 (复用现有实现)
        self.ff = FeedForward(
            in_dim=d_model,
            dim_feedforward=dim_feedforward,
            batch_norm=False,
            activation=activation_type,
            dropout=ffn_dropout
        )
        
        # 条件化 LayerNorm
        self.cond_ln1 = CondLayerNorm(d_model, cond_dim, eps=norm_eps)
        self.cond_ln2 = CondLayerNorm(d_model, cond_dim, eps=norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.debug = debug
        
        self.__name__ = f'Conditional{attention_type.capitalize()}EncoderLayer'
    
    def forward(self, x, cond, pos=None, weight=None):
        """
        Args:
            x: [B, N, d_model] - 输入 token
            cond: [B, N, cond_dim] - 条件 token
            pos: [B, N, pos_dim] - 位置信息 (可选，通常不用因为 pos 已在 cond 中)
            weight: [B, N, N] 或 [B, N] - 质量矩阵 (可选)
        
        Returns:
            [B, N, d_model] - 输出 token
        """
        # PostNorm 风格: 先残差，再 CondLN
        
        # Attention block
        if pos is not None and self.pos_dim > 0:
            att_output, _ = self.attn(x, x, x, pos=pos, weight=weight)
        else:
            att_output, _ = self.attn(x, x, x, weight=weight)
        
        x = self.cond_ln1(x + self.dropout1(att_output), cond)
        
        # FeedForward block
        x1 = self.ff(x)
        x = self.cond_ln2(x + self.dropout2(x1), cond)
        
        return x


class DualHeadFourierTransformer(nn.Module):
    """
    双头条件化 Fourier Transformer
    
    设计思想:
        - G (条件场，不可叠加): 包含物理尺度/pos 信息，用于调制 U 的表征
        - U (源项场，可叠加): 待编码的场变量
        - 通过 CondLayerNorm (FiLM 机制) 在每层 encoder 中用 G 调制 U
        - 不使用 cross-attention，不将 G/U concat
        - pos 信息已编码在 G 的通道中，不额外添加 positional encoding
    
    架构流程:
        1. G/U 分别 embedding: Linear(G_dim -> hidden), Linear(U_dim -> hidden)
        2. Flatten 为 token: [B, H, W(, D), C] -> [B, N, hidden]
        3. Conditional Encoding: 只编码 u_tok，每层用 g_tok 作为 cond
        4. Reshape + Regressor: [B, N, hidden] -> [B, H, W(, D), hidden] -> [B, H, W(, D), C_out]
    
    注意:
        本模型使用 channel-last 格式 [B, H, W(, D), C]，与项目默认数据格式一致。
    
    Args:
        G_dim: G 的通道数 (含 pos 信息)
        U_dim: U 的通道数
        n_targets: 输出通道数
        n_hidden: token 的隐藏维度
        num_encoder_layers: encoder 层数
        n_head: 注意力头数
        dim_feedforward: FeedForward 中间维度
        attention_type: 注意力类型 ('fourier', 'galerkin', 'linear', 'softmax')
        decoder_type: 解码器类型 ('pointwise', 'ifft')
        num_regressor_layers: regressor 层数
        spacial_dim: 空间维度 (2 for 2D, 3 for 3D)
        pos_dim: 位置编码维度 (通常不用，因为 pos 已在 G 中)
        dropout: dropout 比例
        encoder_dropout: encoder 的 dropout
        decoder_dropout: decoder 的 dropout
        ffn_dropout: FeedForward 的 dropout
        activation_type: 激活函数类型
        regressor_activation: regressor 的激活函数
        return_latent: 是否返回中间层特征
        debug: 是否打印调试信息
    
    Input:
        G: [B, H, W(, D), G_dim] - 条件场 (含 pos)，channel 在最后
        U: [B, H, W(, D), U_dim] - 源项场，channel 在最后
        weight: [B, N, N] 或 [B, N] - 质量矩阵 (可选)
        return_latent: 是否返回中间特征
    
    Output:
        T: [B, H, W(, D), n_targets] - 预测场，channel 在最后
        或 dict(preds=T, preds_latent=[...]) 如果 return_latent=True
    """
    
    def __init__(self, **kwargs):
        super(DualHeadFourierTransformer, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self._get_setting()
        self._initialize()
        self.__name__ = f'DualHead{self.attention_type.capitalize()}Transformer'
    
    def forward(self, G, U, weight=None, return_latent=False):
        """
        Args:
            G: [B, H, W(, D), G_dim] - 条件场 (含 pos 信息)，channel 在最后
            U: [B, H, W(, D), U_dim] - 源项场，channel 在最后
            weight: [B, N, N] 或 [B, N] - 质量矩阵 (可选)
            return_latent: 是否返回中间层特征
        
        Returns:
            T: [B, H, W(, D), n_targets] - 预测场，channel 在最后
            或 dict 如果 return_latent=True
        """
        # 形状检查：输入格式为 [B, H, W(, D), C]
        assert G.shape[0] == U.shape[0], f"Batch size mismatch: G{G.shape} vs U{U.shape}"
        assert G.shape[1:-1] == U.shape[1:-1], f"Spatial shape mismatch: G{G.shape} vs U{U.shape}"
        
        bsz = G.size(0)
        spatial_shape = G.shape[1:-1]  # (H, W) or (H, W, D)
        
        # DEBUG: 打印输入形状
        if self.debug:
            print(f"[DualHead] Input G.shape={G.shape}, U.shape={U.shape}")
        
        # 1. Embedding + Flatten
        # 输入格式: [B, H, W(, D), C]，直接 flatten spatial dimensions
        N = 1
        for s in spatial_shape:
            N *= s
        G = G.reshape(bsz, N, -1)  # [B, N, G_dim]
        U = U.reshape(bsz, N, -1)  # [B, N, U_dim]
        
        # Embedding
        g_tok = self.g_downscaler(G)  # [B, N, hidden]
        u_tok = self.u_downscaler(U)  # [B, N, hidden]
        
        if self.debug:
            print(f"[DualHead] g_tok.shape={g_tok.shape}, u_tok.shape={u_tok.shape}")
        
        # 2. Conditional Encoding
        # 只编码 u_tok，每层用 g_tok 作为条件
        x_latent = []
        if self.return_latent:
            x_latent.append(u_tok.contiguous())
        
        u_tok = self.dpo(u_tok)
        
        for encoder in self.encoder_layers:
            u_tok = encoder(u_tok, cond=g_tok, pos=None, weight=weight)
            if self.return_latent:
                x_latent.append(u_tok.contiguous())
        
        # 3. Reshape to spatial
        if self.spacial_dim == 2:
            H, W = spatial_shape
            u_tok = u_tok.view(bsz, H, W, self.n_hidden)
        elif self.spacial_dim == 3:
            H, W, D = spatial_shape
            u_tok = u_tok.view(bsz, H, W, D, self.n_hidden)
        
        if self.debug:
            print(f"[DualHead] After reshape u_tok.shape={u_tok.shape}")
        
        # 4. Regressor
        u_tok = self.dpo(u_tok)
        
        # 注意: 对于 DualHeadTransformer，pos 信息已编码在 G 中
        # 因此通常设置 spacial_fc=False，此时 grid=None 不会被使用
        if self.return_latent:
            out, reg_latent = self.regressor(u_tok, grid=None)
            if reg_latent is not None:
                x_latent.append(reg_latent)
        else:
            out = self.regressor(u_tok, grid=None)
        
        # 输出格式保持 [B, H, W(, D), C] (channel last)
        # 不做 permute，与输入格式一致
        
        if self.debug:
            print(f"[DualHead] Output out.shape={out.shape}")
        
        if return_latent or self.return_latent:
            return dict(preds=out, preds_latent=x_latent)
        else:
            return out
    
    def _initialize(self):
        """初始化各个组件"""
        self._get_embeddings()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)
    
    def _get_setting(self):
        """获取配置参数"""
        # 必需参数检查
        required_params = ['G_dim', 'U_dim', 'n_targets']
        missing_params = []
        for param in required_params:
            if param not in self.config or self.config[param] is None:
                missing_params.append(param)
        
        if missing_params:
            # 提供详细的错误信息
            error_msg = f"\nDualHeadFourierTransformer requires the following parameters:\n"
            error_msg += f"  - G_dim: number of channels in G (condition field, including pos)\n"
            error_msg += f"  - U_dim: number of channels in U (source field)\n"
            error_msg += f"  - n_targets: number of output channels\n"
            error_msg += f"\nMissing or None parameters: {missing_params}\n"
            error_msg += f"\nNote: This model uses a different interface than FourierTransformer.\n"
            error_msg += f"      Instead of 'node_feats', you need to specify 'G_dim' and 'U_dim' separately.\n"
            error_msg += f"\nExample configuration:\n"
            error_msg += f"  config = dict(\n"
            error_msg += f"      G_dim=2,           # G has 2 channels (e.g., pos_x, pos_y)\n"
            error_msg += f"      U_dim=3,           # U has 3 channels (source field)\n"
            error_msg += f"      n_targets=1,       # output 1 channel (e.g., temperature)\n"
            error_msg += f"      n_hidden=96,\n"
            error_msg += f"      num_encoder_layers=4,\n"
            error_msg += f"      n_head=4,\n"
            error_msg += f"      attention_type='fourier',\n"
            error_msg += f"      decoder_type='pointwise',\n"
            error_msg += f"      spacial_dim=2,\n"
            error_msg += f"  )\n"
            error_msg += f"\nCurrent config keys: {list(self.config.keys())}\n"
            raise ValueError(error_msg)
        
        self.G_dim = self.config['G_dim']
        self.U_dim = self.config['U_dim']
        self.n_targets = self.config['n_targets']
        
        # 可选参数
        self.n_hidden = default(self.config.get('n_hidden'), 96)
        self.num_encoder_layers = default(self.config.get('num_encoder_layers'), 4)
        self.n_head = default(self.config.get('n_head'), 4)
        self.dim_feedforward = default(self.config.get('dim_feedforward'), 2 * self.n_hidden)
        self.attention_type = default(self.config.get('attention_type'), 'fourier')
        self.decoder_type = default(self.config.get('decoder_type'), 'pointwise')
        self.num_regressor_layers = default(self.config.get('num_regressor_layers'), 2)
        self.spacial_dim = default(self.config.get('spacial_dim'), 2)
        self.pos_dim = default(self.config.get('pos_dim'), 0)  # 通常为 0，因为 pos 已在 G 中
        
        # Dropout
        self.dropout = default(self.config.get('dropout'), 0.05)
        self.encoder_dropout = default(self.config.get('encoder_dropout'), self.dropout)
        self.decoder_dropout = default(self.config.get('decoder_dropout'), self.dropout)
        self.ffn_dropout = default(self.config.get('ffn_dropout'), self.dropout)
        self.dpo = nn.Dropout(self.dropout)
        
        # 激活函数
        self.activation_type = default(self.config.get('activation_type'), 'silu')
        self.regressor_activation = default(self.config.get('regressor_activation'), 'silu')
        
        # 注意力参数
        self.xavier_init = default(self.config.get('xavier_init'), 1e-2)
        self.diagonal_weight = default(self.config.get('diagonal_weight'), 1e-2)
        self.symmetric_init = default(self.config.get('symmetric_init'), False)
        self.norm_eps = default(self.config.get('norm_eps'), 1e-5)
        
        # Regressor 参数
        self.freq_dim = default(self.config.get('freq_dim'), 64)
        self.fourier_modes = default(self.config.get('fourier_modes'), 16)
        self.spacial_fc = default(self.config.get('spacial_fc'), False)
        
        # 其他
        self.return_latent = default(self.config.get('return_latent'), False)
        self.debug = default(self.config.get('debug'), False)
    
    def _get_embeddings(self):
        """创建 G 和 U 的 embedding 层"""
        # G embedding: G_dim -> n_hidden
        self.g_downscaler = nn.Linear(self.G_dim, self.n_hidden)
        
        # U embedding: U_dim -> n_hidden
        self.u_downscaler = nn.Linear(self.U_dim, self.n_hidden)
    
    def _get_encoder(self):
        """创建条件化 encoder 层"""
        encoder_layer = ConditionalTransformerEncoderLayer(
            d_model=self.n_hidden,
            cond_dim=self.n_hidden,  # g_tok 和 u_tok 维度相同
            pos_dim=self.pos_dim,
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            attention_type=self.attention_type,
            xavier_init=self.xavier_init,
            diagonal_weight=self.diagonal_weight,
            symmetric_init=self.symmetric_init,
            activation_type=self.activation_type,
            dropout=self.encoder_dropout,
            ffn_dropout=self.ffn_dropout,
            norm_eps=self.norm_eps,
            debug=self.debug
        )
        
        self.encoder_layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)
        ])
    
    def _get_regressor(self):
        """创建 regressor (解码器)"""
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(
                in_dim=self.n_hidden,
                n_hidden=self.n_hidden,
                out_dim=self.n_targets,
                num_layers=self.num_regressor_layers,
                spacial_fc=self.spacial_fc,
                spacial_dim=self.spacial_dim,
                activation=self.regressor_activation,
                dropout=self.decoder_dropout,
                return_latent=self.return_latent,
                debug=self.debug
            )
        elif self.decoder_type == 'ifft':
            self.regressor = SpectralRegressor(
                in_dim=self.n_hidden,
                n_hidden=self.n_hidden,
                freq_dim=self.freq_dim,
                out_dim=self.n_targets,
                num_spectral_layers=self.num_regressor_layers,
                modes=self.fourier_modes,
                spacial_dim=self.spacial_dim,
                spacial_fc=self.spacial_fc,
                activation=self.regressor_activation,
                dropout=self.decoder_dropout,
                return_latent=self.return_latent,
                debug=self.debug
            )
        else:
            raise NotImplementedError(f"Decoder type {self.decoder_type} not implemented")
    
    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print(f"DualHeadFourierTransformer Configuration:")
        print("=" * 50)
        for key, value in self.config.items():
            if not key.startswith('__'):
                print(f"{key:30s}: {value}")
        print("=" * 50)


if __name__ == '__main__':
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2D 测试
    config = dict(
        G_dim=2,           # G 含 2 个通道 (pos_x, pos_y)
        U_dim=3,           # U 含 3 个通道 (源项场)
        n_targets=1,       # 输出 1 个通道 (温度场)
        n_hidden=96,
        num_encoder_layers=4,
        n_head=4,
        attention_type='fourier',
        decoder_type='pointwise',
        spacial_dim=2,
        dropout=0.05,
        debug=True,
    )
    
    model = DualHeadFourierTransformer(**config)
    model.to(device)
    model.print_config()
    
    # 测试前向传播 (channel-last 格式)
    batch_size = 4
    H, W = 32, 32
    G = torch.randn(batch_size, H, W, 2).to(device)  # [B, H, W, C]
    U = torch.randn(batch_size, H, W, 3).to(device)  # [B, H, W, C]
    
    print(f"\nInput (channel-last): G.shape={G.shape}, U.shape={U.shape}")
    
    out = model(G, U)
    print(f"Output (channel-last): out.shape={out.shape}")
    
    assert out.shape == (batch_size, H, W, 1), f"Output shape mismatch: {out.shape}"
    print("\n✓ 2D test passed (channel-last format)!")

