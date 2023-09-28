from basic.basic_layers import *
from Models.configs import *


class DeepONetMulti(nn.Layer):
    # =============================================================================
    #     Inspired by L. Lu, J. Pengzhan, G.E. Karniadakis.
    #     "DeepONet: Learning nonlinear operators for identifying differential equations based on
    #     the universal approximation theorem of operators".
    #     arXiv:1910.03193v3 [cs.LG] 15 Apr 2020.
    # =============================================================================
    def __init__(self, input_dim: int, operator_dims: list, output_dim: int,
                 planes_branch: list, planes_trunk: list, activation='gelu'):
        """
        :param input_dim: int, the coordinates dim for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param output_dim: int, the predicted variable dims
        :param planes_branch: list, the hidden layers dims for branch net
        :param planes_trunk: list, the hidden layers dims for trunk net
        :param operator_dims: list，the operate dims list for each branch net
        :param activation: activation function
        """
        super(DeepONetMulti, self).__init__()

        self.branches = nn.LayerList() # 分支网络
        self.trunks = nn.LayerList() # 主干网络
        for dim in operator_dims:
            self.branches.append(FcnSingle([dim] + planes_branch, activation=activation))# FcnSingle是从basic_layers里导入的
        for _ in range(output_dim):
            self.trunks.append(FcnSingle([input_dim] + planes_trunk, activation=activation))

        self.reset_parameters()

    def reset_parameters(self): # 初始化所有网络的参数
        """
        weight initialize
        """
        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight, gain=1)
                nn.initializer.XavierUniform(m.weight)
                # m.bias.data.zero_()

    def forward(self, u_vars, y_var, size_set=True):
        """
        forward compute
        :param u_vars: tensor list[(batch_size, ..., operator_dims[0]), (batch_size, ..., operator_dims[1]), ...]
        :param y_var: (batch_size, ..., input_dim)
        :param size_set: bool, true for standard inputs, false for reduce points number in operator inputs
        """
        B = 1.
        for u_var, branch in zip(u_vars, self.branches):
            B *= branch(u_var)
        if not size_set:
            B_size = list(y_var.shape[1:-1])
            for i in range(len(B_size)):
                B = B.unsqueeze(1)
            B = paddle.tile(B, [1, ] + B_size + [1, ])

        out_var = []
        for trunk in self.trunks:
            T = trunk(y_var)
            out_var.append(paddle.sum(B * T, axis=-1)) # 用这种方式实现两个网络的乘积
        out_var = paddle.stack(out_var, axis=-1)
        return out_var


if __name__ == "__main__":
    us = [paddle.ones([10, 256 * 2]), paddle.ones([10, 1])]
    x = paddle.ones([10, 2])
    layer = DeepONetMulti(input_dim=2, operator_dims=[256 * 2, 1], output_dim=5,
                          planes_branch=[64] * 3, planes_trunk=[64] * 2)
    y = layer(us, x)
    print(y.shape)
