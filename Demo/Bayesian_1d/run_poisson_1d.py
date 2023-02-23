#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/2/20 10:16
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @Site    : 
# @File    : run_poisson_1d.py
"""

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

import Utilizes.util as util
from Utilizes.visual_data import MatplotlibVision, TextLogger
import Bayesian_util as Bayesian
from basic.basic_layers import FcnSingle

import os
import sys

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# hyperparameters

util.set_random_seed(123)
prior_std = 1
like_std = 0.1
step_size = 0.001
burn = 100
num_samples = 200
L = 100
layer_planes = [1, 16, 16, 1]
pde = True
pinns = False
epochs = 10000
tau_priors = 1 / prior_std ** 2
tau_likes = 1 / like_std ** 2

lb = -0.7
ub = 0.7
N_tr_u = 2
N_tr_f = 16
N_val = 100


# data

def u(x):
    return torch.sin(6 * x) ** 3


def f(x):
    return 0.01 * (-108) * torch.sin(6 * x) * (torch.sin(6 * x) ** 2 - 2 * torch.cos(6 * x) ** 2)


data = {}
data['x_u'] = torch.linspace(lb, ub, N_tr_u).view(-1, 1)
data['y_u'] = u(data['x_u']) + torch.randn_like(data['x_u']) * like_std  # +生成均值为0，方差为0.1的数据
data['x_f'] = torch.linspace(lb, ub, N_tr_f).view(-1, 1)
data['y_f'] = f(data['x_f']) + torch.randn_like(data['x_f']) * like_std  # +生成均值为0，方差为0.1的数据

data_val = {}
data_val['x_u'] = torch.linspace(lb, ub, N_val).view(-1, 1)
data_val['y_u'] = u(data_val['x_u'])
data_val['x_f'] = torch.linspace(lb, ub, N_val).view(-1, 1)
data_val['y_f'] = f(data_val['x_f'])

for d in data:
    data[d] = data[d].to(device)
for d in data_val:
    data_val[d] = data_val[d].to(device)

# model

name = 'poisson-1d'
work_path = os.path.join('work', name)
isCreated = os.path.exists(work_path)
if not isCreated:
    os.makedirs(work_path)

# 将控制台的结果输出到a.log文件，可以改成a.txt
sys.stdout = TextLogger(os.path.join(work_path, 'train.log'), sys.stdout)


# net_u = FcnSingle(layer_planes, activation='gelu').to(device)

class Net(nn.Module):

    def __init__(self, layer_sizes, activation=torch.tanh):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        # self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        # x = self.activation(x)
        # x = self.l4(x)
        return x


activation = torch.tanh
net_u = Net(layer_planes, activation).to(device)
nets = [net_u]


# net_u = FcnSingle(layer_planes, activation='tanh').to(device)
# nets = [net_u_]


def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    x_u = data['x_u']
    y_u = data['y_u']
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data['x_f']
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]
    pred_f = 0.01 * u_xx
    y_f = data['y_f']
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


# sampling

params_hmc = Bayesian.sample_model_bpinns(nets, data, model_loss=model_loss, num_samples=num_samples,
                                          num_steps_per_sample=L, step_size=step_size, burn=burn, tau_priors=tau_priors,
                                          tau_likes=tau_likes, device=device, pde=pde, pinns=pinns, epochs=epochs)

pred_list, log_prob_list = Bayesian.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss,
                                                         tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

pred_list_u = pred_list[0].cpu().numpy()
pred_list_f = pred_list[1].cpu().numpy()

# plot

x_val = data_val['x_u'].cpu().numpy()
u_val = data_val['y_u'].cpu().numpy()
f_val = data_val['y_f'].cpu().numpy()
x_u = data['x_u'].cpu().numpy()
y_u = data['y_u'].cpu().numpy()
x_f = data['x_f'].cpu().numpy()
y_f = data['y_f'].cpu().numpy()

plt.figure(figsize=(7, 5))
plt.plot(x_val, u_val, 'r-', label='Exact')
# plt.plot(x_val,pred_list_u.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val, pred_list_u.mean(0).squeeze().T, 'b-', alpha=0.9, label='Mean')
plt.fill_between(x_val.reshape(-1), pred_list_u.mean(0).squeeze().T - 2 * pred_list_u.std(0).squeeze().T,
                 pred_list_u.mean(0).squeeze().T + 2 * pred_list_u.std(0).squeeze().T, facecolor='b', alpha=0.2,
                 label='2 std')
plt.plot(x_u, y_u, 'kx', markersize=5, label='Training data')
plt.xlim([lb, ub])
plt.legend(fontsize=10)
plt.savefig(os.path.join(work_path, 'u_predict.jpg'))

plt.figure(figsize=(7, 5))
plt.plot(x_val, f_val, 'r-', label='Exact')
# plt.plot(x_val,pred_list_f.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val, pred_list_f.mean(0).squeeze().T, 'b-', alpha=0.9, label='Mean')
plt.fill_between(x_val.reshape(-1), pred_list_f.mean(0).squeeze().T - 2 * pred_list_f.std(0).squeeze().T,
                 pred_list_f.mean(0).squeeze().T + 2 * pred_list_f.std(0).squeeze().T, facecolor='b', alpha=0.2,
                 label='2 std')
plt.plot(x_f, y_f, 'kx', markersize=5, label='Training data')
plt.xlim([lb, ub])
plt.legend(fontsize=10)
plt.savefig(os.path.join(work_path, 'f_predict.jpg'))
# %%
