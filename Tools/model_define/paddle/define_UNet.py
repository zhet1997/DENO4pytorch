
# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/4/17 22:06
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py
@File ：run_FNO.py
"""
import os
import paddle
from train_model.train_task_construct import feature_transform


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
    for batch, (xx, yy) in enumerate(dataloader):
        gd = feature_transform(xx)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy)

        optimizer.clear_grad()
        loss.backward()
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
        for batch, (xx, yy) in enumerate(dataloader):
            gd = feature_transform(xx)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy)
            valid_loss += loss.item()

    return valid_loss / (batch + 1)


def inference(dataloader, netmodel, device): # 这个是？？
    """
    Args:
        dataloader: input coordinates
        netmodel: Network
    Returns:
        out_pred: predicted fields
    """

    with paddle.no_grad():
        xx, yy = next(iter(dataloader))
        gd = feature_transform(xx)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), gd.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################

    name = 'UNet'
