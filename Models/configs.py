import paddle.nn as nn


activation_dict = \
    {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'leakyrelu': nn.LeakyReLU(),
     None: nn.GELU()}

additional_attr = ['normalizer', 'raw_laplacian', 'return_latent',
                   'residual_type', 'norm_type', 'norm_eps', 'boundary_condition',
                   'upscaler_size', 'downscaler_size', 'spacial_dim', 'spacial_fc',
                   'regressor_activation', 'attn_activation',
                   'downscaler_activation', 'upscaler_activation',
                   'encoder_dropout', 'decoder_dropout', 'ffn_dropout']


def default(value, d):
    """
        helper taken from https://github.com/lucidrains/linear-attention-transformer
    """
    return d if value is None else value
