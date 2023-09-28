import math
import numpy as np
import paddle
from Models.configs import *
from typing import Any, List, Tuple, Union

class FcnSingle(nn.Layer):
    def __init__(self, planes: list or tuple, activation="gelu", last_activation=False):
        # =============================================================================
        #     Inspired by M. Raissi a, P. Perdikaris b,∗, G.E. Karniadakis.
        #     "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems
        #     involving nonlinear partial differential equations".
        #     Journal of Computational Physics.
        # =============================================================================
        super(FcnSingle, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.LayerList()
        for i in range(len(self.planes) - 2):
            self.layers.append(nn.Linear(self.planes[i], self.planes[i + 1]))
            self.layers.append(self.active)
        self.layers.append(nn.Linear(self.planes[-2], self.planes[-1]))

        if last_activation:
            self.layers.append(self.active)
        self.layers = nn.Sequential(*self.layers)  # *的作用是解包

        # self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.initializer.XavierUniform(m.weight)#这里存疑
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        out_var = self.layers(in_var)
        return out_var


class FcnMulti(nn.Layer):
    def __init__(self, planes: list, activation="gelu"):
        # =============================================================================
        #     Inspired by Haghighat Ehsan, et all.
        #     "A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics"
        #     Computer Methods in Applied Mechanics and Engineering.
        # =============================================================================
        super(FcnMulti, self).__init__()
        self.planes = planes
        self.active = activation_dict[activation]

        self.layers = nn.LayerList()
        for j in range(self.planes[-1]):
            layer = []
            for i in range(len(self.planes) - 2):
                layer.append(nn.Linear(self.planes[i], self.planes[i + 1]))
                layer.append(self.active)
            layer.append(nn.Linear(self.planes[-2], 1))
            self.layers.append(nn.Sequential(*layer))
        self.reset_parameters()

    def reset_parameters(self):
        """
        weight initialize
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.initializer.XavierUniform(m.weight)#这里存疑
                m.bias.data.zero_()

    def forward(self, in_var):
        """
        forward compute
        :param in_var: (batch_size, ..., input_dim)
        """
        y = []
        for i in range(self.planes[-1]):
            y.append(self.layers[i](in_var))
        return paddle.concat(y, axis=-1)


class Identity(nn.Layer):
    '''
    a placeholder layer similar to tensorflow.no_op():
    https://github.com/pypaddle/pypaddle/issues/9160#issuecomment-483048684
    not used anymore as
    https://pypaddle.org/docs/stable/generated/paddle.nn.Identity.html
    edge and grid are dummy inputs
    '''

    def __init__(self, in_features=None, out_features=None,
                 *args, **kwargs):
        super(Identity, self).__init__()

        if in_features is not None and out_features is not None:
            self.id = nn.Linear(in_features, out_features)

        else:
            self.id = nn.Identity()

    def forward(self, x, edge=None, grid=None):
        """
        forward compute
        :param in_var: (batch_size, input_dim, ...)
        """
        # todo: 利用 einsteinsum 构造
        if len(x.shape) == 5:
            '''
            (-1, in, H, W, S) -> (-1, out, H, W, S)
            Used in SimpleResBlock
            '''
            x = x.transpose([0, 2, 3, 4, 1])
            x = self.id(x)
            x = x.transpose([0, 4, 1, 2, 3])
        elif len(x.shape) == 4:
            '''
            (-1, in, H, W) -> (-1, out, H, W)
            Used in SimpleResBlock
            '''
            x = x.transpose([0, 2, 3, 1])
            x = self.id(x)
            x = x.transpose([0, 3, 1, 2])

        elif len(x.shape) == 3:
            '''
            (-1, S, in) -> (-1, S, out)
            Used in SimpleResBlock
            '''
            # x = x.transpose([0, 2, 1])
            x = self.id(x)
            # x = x.transpose([0, 2, 1])
        elif len(x.shape) == 2:
            '''
            (-1, in) -> (-1, out)
            Used in SimpleResBlock
            '''
            x = self.id(x)
        else:
            raise NotImplementedError("input sizes not implemented.")

        return x


class Empircal(object):
    """
    Empirical model
    """
    def __init__(self, parameters: Union[dict, None] = None, models: Union[dict, None] = None):
        super().__init__()
        self._parameters = parameters
        self.__dict__.update(parameters)
        self._modules = models
        self.__dict__.update(models)


    def train(self, x: Union[np.ndarray, None] = None,
                    h: Union[np.ndarray, None] = None,
                    y: Union[np.ndarray, None] = None,
              ) -> np.ndarray:
        """
        :param x: input data
        :param h: auxiliary data
        :param y: output data
        :return:
        """

    def infer(self, x: Union[np.ndarray, None] = None,
                    h: Union[np.ndarray, None] = None,
                    y: Union[np.ndarray, None] = None,
                    steps_ahead: Union[int, None] = None,
                    forecast_horizon: Union[int, None] = None,
                    **kwargs
              ) -> np.ndarray:
        """
        infer the model
        :param x: input data
        :param h: auxiliary data
        :param y: output initial data
        :param steps_ahead:
        :param forecast_horizon:
        :return:
        """

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)


if __name__ == '__main__':
    x = paddle.ones([10, 64, 64, 3])
    layer = FcnSingle([3, 64, 64, 10])
    y = layer(x)
    print(y.shape)

    x = paddle.ones([10, 64, 64, 3])
    layer = FcnMulti([3, 64, 64, 10])
    y = layer(x)
    print(y.shape)

