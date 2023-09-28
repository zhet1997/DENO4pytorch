import paddle
import paddle.nn as nn


from functools import partial
from Models.configs import activation_dict


class SpectralConv1d(nn.Layer):
    '''
    1维谱卷积
    Modified Zongyi Li's Spectral1dConv code
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_1d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: int,  # number of fourier modes
                 dropout=0.1,
                 norm="ortho",
                 return_freq=False,
                 activation='gelu'):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.activation = activation_dict[activation]
        self.linear = nn.Conv1D(self.in_dim, self.out_dim, 1)  # for residual
        # self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = paddle.create_parameter(self.scale * paddle.rand(in_dim, out_dim, self.modes, dtype='float32'))
        # xavier_normal_(self.weights1, gain=1 / (in_dim * out_dim))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return paddle.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        forward computation
        """
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.linear(x)
        # x = self.dropout(x)
        x_ft = paddle.fft.rfft(x, norm=self.norm)

        # Multiply relevant Fourier modes
        shape = paddle.to_tensor([batchsize, self.out_dim, x.sshape[-1] // 2 + 1], place=x.place)
        out_ft = paddle.zeros(shape=shape, dtype='float32')
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        # Return to physical space
        x = paddle.fft.irfft(out_ft, norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv2d(nn.Layer):
    '''
    2维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,  # number of fourier modes
                 dropout=0.1,
                 norm='ortho',
                 activation='gelu',
                 return_freq=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.activation = activation_dict[activation]
        self.return_freq = return_freq
        self.linear = nn.Conv2D(self.in_dim, self.out_dim, 1)  # for residual

        self.scale = (1 / (in_dim * out_dim))
        weights1 = paddle.to_tensor(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2]), dtype='float32')
        self.weights1 = paddle.create_parameter(shape=weights1.shape,
                                               default_initializer=paddle.nn.initializer.Assign(weights1),
                                               dtype='float32')
        weights2 = paddle.to_tensor(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2]), dtype='float32')
        self.weights2 = paddle.create_parameter(shape=weights2.shape,
                                               default_initializer=paddle.nn.initializer.Assign(weights2),
                                               dtype='float32')
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        temp = paddle.einsum("bixy,ioxy->boxy", input, weights)
        return paddle.cast(temp, dtype='float32')

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.linear(x)
        x = self.dropout(x)
        x_ft = paddle.fft.rfft2(x, norm=self.norm)

        # Multiply relevant Fourier modes
        shape = paddle.to_tensor([batch_size, self.out_dim, x.shape[-2], x.shape[-1] // 2 + 1], place=x.place)
        out_ft = paddle.zeros(shape=shape, dtype='float32')
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = paddle.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


class SpectralConv3d(nn.Layer):
    '''
    三维谱卷积
    Modified Zongyi Li's SpectralConv2d PyTorch 1.6 code
    using only real weights
    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py
    '''

    def __init__(self, in_dim,
                 out_dim,
                 modes: tuple,
                 dropout=0.1,
                 norm='ortho',
                 activation='silu',
                 return_freq=False):  # whether to return the frequency target
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(modes, int):
            self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes
            self.modes3 = modes
        else:
            self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
            self.modes2 = modes[1]
            self.modes3 = modes[2]

        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.return_freq = return_freq
        self.activation = activation_dict[activation]

        self.linear = nn.Conv3D(self.in_dim, self.out_dim, 1)  # for residual

        self.scale = (1 / (in_dim * out_dim))
        self.weights1 = paddle.create_parameter(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2, self.modes3],
                                    dtype='float32'))
        self.weights2 = paddle.create_parameter(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2, self.modes3],
                                    dtype='float32'))
        self.weights3 = paddle.create_parameter(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2, self.modes3],
                                    dtype='float32'))
        self.weights4 = paddle.create_parameter(
            self.scale * paddle.rand([in_dim, out_dim, self.modes1, self.modes2, self.modes3],
                                    dtype='float32'))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        forward computation
        """
        batch_size = x.size(0)
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        res = self.linear(x)
        # x = self.dropout(x)
        x_ft = paddle.fft.rfftn(x, axes=[-3, -2, -1], norm=self.norm)
        # Multiply relevant Fourier modes
        shape = paddle.to_tensor([batch_size, self.out_dim, x.shape[-3], x.shape[-2], x.shape[-1] // 2 + 1], place=x.place)
        out_ft = paddle.zeros(shape=shape, dtype='float32')
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = paddle.fft.irfftn(out_ft, s=(x.shape[-3], x.shape[-2], x.shape[-1]), norm=self.norm)
        x = self.activation(x + res)

        if self.return_freq:
            return x, out_ft
        else:
            return x


if __name__ == '__main__':
    x = paddle.ones([10, 3, 64])
    layer = SpectralConv1d(in_dim=3, out_dim=10, modes=5)
    y = layer(x)
    print(y.shape)

