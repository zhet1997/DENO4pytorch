# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/4/17 22:06
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py
@File ：run_FNO.py
"""
import numpy as np
import paddle
import paddle.nn as nn
from Tools.visualization import MatplotlibVision
from Tools.data_process import DataNormer
import matplotlib.pyplot as plt
import time
import os
from Tools.utilizes_rotor37 import get_grid, get_origin

class MLP(nn.Layer):
    def __init__(self, layers=None, is_BatchNorm=True,
                 in_dim=None,
                 out_dim=None,
                 n_hidden=None,
                 num_layers=None):
        if layers is None:
            layers = [in_dim]
            for ii in range(num_layers-2):
                layers.append(n_hidden)
            layers.append(out_dim)
        super(MLP, self).__init__()
        self.depth = len(layers)
        self.activation = nn.GELU
        #先写完整的layerslist
        layer_list = []
        for i in range(self.depth-2):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            if is_BatchNorm is True:
                layer_list.append(('batchnorm_%d' % i, nn.BatchNorm1D(layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        #最后一层，输出层
        layer_list.append(('layer_%d' % (self.depth-2), nn.Linear(layers[-2], layers[-1])))
        # layerDict = OrderedDict(layer_list)
        layerDict = layer_list
        #再直接使用sequential生成网络
        self.layers = nn.Sequential(*layerDict)

    def forward(self,x):
        y = self.layers(x)
        return y

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    """
    Args:
        data_loader: output fields at last time step
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    train_loss = 0
    for batch, (input,output) in enumerate(dataloader):
        # input = input.to(device)
        # output = output.to(device)
        pred = netmodel(input)
        loss = lossfunc(pred, output)
        optimizer.clear_grad()
        loss.backward() # 自动微分
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def valid(dataloader, netmodel, device, lossfunc):
    """
    Args:
        data_loader: input coordinates
        model: Network
        lossfunc: Loss function
    """
    valid_loss = 0
    with paddle.no_grad():
        for batch, (input, output) in enumerate(dataloader):
            pred = netmodel(input)
            loss = lossfunc(pred, output)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)

def inference(dataloader, netmodel, device):
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """
    with paddle.no_grad():
        xx, yy = next(iter(dataloader))
        pred = netmodel(xx)

    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
################################################################
# configs
################################################################
    name = 'MLP'
    work_path = os.path.join('work', name)
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    if paddle.device.is_compiled_with_cuda():
        Device = paddle.set_device('gpu')
    else:
        Device = paddle.set_device('cpu')

    in_dim = 28
    out_dim = 5

    ntrain = 2700
    nvalid = 250

    batch_size = 32
    epochs = 1001

    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

################################################################
# load data
################################################################

    design, fields = get_origin(realpath=os.path.join("..","data"))
    # design = get_gemodata()

    input = design
    input = paddle.to_tensor(input, dtype='float32')
    output = fields
    output = paddle.to_tensor(output, dtype='float32')
    print(input.shape, output.shape)

    train_x = input[:ntrain, :]
    train_y = output[:ntrain, :]
    valid_x = input[-nvalid:, :]
    valid_y = output[-nvalid:, :]

    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.save(os.path.join(work_path, 'x_norm.pkl')) # 将normalizer保存下来
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    train_y = train_y.reshape([train_x.shape[0],-1])
    valid_y = valid_y.reshape([valid_x.shape[0],-1])

    train_loader = paddle.io.DataLoader(paddle.io.TensorDataset([train_x, train_y]),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = paddle.io.DataLoader(paddle.io.TensorDataset([valid_x, valid_y]),
                                               batch_size=batch_size, shuffle=False, drop_last=True)


################################################################
# Neural Networks
################################################################

    # 建立网络
    layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim*64*64]
    Net_model = MLP(layers=layer_mat, is_BatchNorm=False)
    Net_model = Net_model.to(Device)
    print(name)

# 损失函数
    Loss_func = nn.MSELoss()
    # Loss_func = nn.SmoothL1Loss()
    # 优化算法
    Scheduler = paddle.optimizer.lr.StepDecay(learning_rate, step_size=scheduler_step, gamma=scheduler_gamma)
    Optimizer = paddle.optimizer.Momentum(parameters=Net_model.parameters(),learning_rate=learning_rate, weight_decay=1e-4)
    # 下降策略
    # Scheduler = paddle.optimizer.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('Ps', 'Ts', 'rhoV', 'Pt', 'Tt'))

    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################
    grid = get_grid(realpath=os.path.join("..","data"))

    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, Optimizer.get_lr(), log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

    ################################################################
    # Visualization
    ################################################################

        if epoch > 0 and epoch % 5 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work_path, 'log_loss.svg'))
            plt.close(fig)


