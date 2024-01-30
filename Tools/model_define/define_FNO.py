
# -*- coding: utf-8 -*-
"""
# @copyright (c) 2023 Baidu.com, Inc. Allrights Reserved
@Time ： 2023/4/17 22:06
@Author ： Liu Tianyuan (liutianyuan02@baidu.com)
@Site ：run_FNO.py
@File ：run_FNO.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from fno.FNOs import FNO2d

def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape([1, size_x, 1, 1]).tile([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape([1, 1, size_y, 1]).tile([batchsize, size_x, 1, 1])
    return torch.concat((gridx, gridy), axis=-1)

def train(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx)
        gd = gd.to(device)
        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy, xx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)

def train_record(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)
        gd = feature_transform(xx)
        gd = gd.to(device)
        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy, xx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)


def train_random(dataloader, netmodel, device, lossfunc, optimizer, scheduler):
    train_loss = 0
    for batch, (xx, yy) in enumerate(dataloader):
        xx = xx.to(device)
        yy = yy.to(device)

        for ii in range(xx.shape[0]):
            idx = torch.randperm(xx.shape[-1])
            xx[ii] = xx[ii,..., idx]

        gd = feature_transform(xx)
        gd = gd.to(device)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy, xx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    return train_loss / (batch + 1)



def train_mask(dataloader, netmodel, device, lossfunc, optimizer, scheduler, x_norm=None):
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
        xx = xx.to(device)
        yy = yy.to(device)

        xx_mask = torch.zeros([*xx.shape[:-1],1], device=device)
        xx_none = torch.zeros([*xx.shape[1:-1],1], device=device) + 350
        xx_none = x_norm.norm(xx_none)
        for ii in range(xx.shape[0]):
            idx = torch.randperm(xx.shape[-1])
            xx[ii] = xx[ii,..., idx]
            xx_mask[ii] = xx[ii, ..., idx[0]:idx[0]+1]
            xx[ii, ..., idx[0]:idx[0]+1] = xx_none.clone().detach()

        gd = feature_transform(xx)
        gd = gd.to(device)

        pred = netmodel(xx, gd)
        loss = lossfunc(pred, yy, xx_mask)

        optimizer.zero_grad()
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
    with torch.no_grad():
        for batch, (xx, yy) in enumerate(dataloader):
            xx = xx.to(device)
            yy = yy.to(device)
            gd = feature_transform(xx)
            gd = gd.to(device)

            pred = netmodel(xx, gd)
            loss = lossfunc(pred, yy, xx)
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

    with torch.no_grad():
        xx, yy = next(iter(dataloader))
        xx = xx.to(device)
        gd = feature_transform(xx)
        gd = gd.to(device)
        pred = netmodel(xx, gd)

    # equation = model.equation(u_var, y_var, out_pred)
    return xx.cpu().numpy(), yy.numpy(), pred.cpu().numpy()


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    pass

