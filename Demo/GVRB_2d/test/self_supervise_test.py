import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Utilizes.process_data import DataNormer
from basic.basic_layers import FcnSingle
from transformer.Transformers import FourierTransformer
from Utilizes.geometrics import gen_uniform_grid
from Utilizes.visual_data import MatplotlibVision, TextLogger
from Demo.GVRB_2d.utilizes_GVRB import GVRBWeightLoss
from fno.FNOs import FNO2d
from cnn.ConvNets import UNet2d

import matplotlib.pyplot as plt
import time
import yaml
from Demo.GVRB_2d.utilizes_GVRB import get_origin, SelfSuperviseLoss, SelfSuperviseLoss2, SelfSuperviseLoss3, SelfSuperviseLoss4
from Demo.GVRB_2d.train_model_GVRB.model_whole_life import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d


def generate_virtual_loader(x_normalizer, virtual_batchs, batch_size,
                            scale=[-0.02, 0.02],
                            in_dim = 100,
                            out_dim=8,
                            sim_multi=1,
                            ):
    half = int(batch_size/2)
    data_virtual = x_normalizer.sample_generate(virtual_batchs*batch_size, 2, norm=False)

    post = cfdPost_2d()
    post.bouCondition_data_readin(
        boundarycondition=data_virtual[:, -4:],
    )

    field_matrix, bc_matrix = post.get_dimensional_matrix(expand=sim_multi, scale=scale)
    field_matrix = np.power(10, field_matrix)


    data_virtual_sim = data_virtual.copy()
    data_virtual_sim[:, -4:] = post.data_similarity_operate(data_virtual[:, -4:].copy(), bc_matrix)
    data_virtual_sim = x_normalizer.norm(data_virtual_sim)
    data_virtual = x_normalizer.norm(data_virtual)

    input_virtual = np.zeros([virtual_batchs*batch_size*2 , in_dim])
    matrix_virtual = np.zeros([virtual_batchs * batch_size * 2, out_dim])
    one_matrix = np.ones_like(field_matrix)


    input_virtual = np.concatenate([data_virtual, data_virtual_sim], axis=-1)
    input_virtual = np.concatenate(
        [input_virtual[i:i + half, :] for i in range(0, virtual_batchs * batch_size, half)], axis=-1)
    input_virtual = np.concatenate(
        [input_virtual[:, i:i + in_dim] for i in range(0, virtual_batchs * in_dim * 4, in_dim)], axis=0)


    matrix_virtual = np.concatenate([one_matrix,field_matrix], axis=-1)
    matrix_virtual = np.concatenate(
        [matrix_virtual[i:i + half, :] for i in range(0, virtual_batchs*batch_size, half)], axis=-1)
    matrix_virtual = np.concatenate(
        [matrix_virtual[:, i:i+out_dim] for i in range(0, virtual_batchs * out_dim * 4, out_dim)], axis=0)

    # for ii in range(virtual_batchs):
    #     input_virtual[ii * batch_size:ii * batch_size + half, :] = data_virtual[ii * half:(ii + 1) * half, :]
    #     input_virtual[ii * batch_size + half:(ii+1) * batch_size , :] = data_virtual_sim[ii * half:(ii + 1) * half, :]
    #
    #     matrix_virtual[ii * batch_size:ii * batch_size + half, :] = np.ones([half, out_dim])
    #     matrix_virtual[ii * batch_size + half:(ii + 1) * batch_size, :] = field_matrix[ii * half:(ii + 1) * half, :]

    matrix_virtual = np.tile(matrix_virtual[:, None, None, :], [1, 64, 128, 1])
    input_virtual = np.tile(input_virtual[:, None, None, :], [1, 64, 128, 1])
    matrix_virtual = torch.as_tensor(matrix_virtual, dtype=torch.float)
    input_virtual = torch.as_tensor(input_virtual, dtype=torch.float)

    self_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(input_virtual, matrix_virtual),
                                              batch_size=batch_size, shuffle=False, drop_last=True)
    return self_loader


def generate_virtual_loader_new(x_normalizer, virtual_batchs, batch_size,
                            scale=[-0.02, 0.02],
                            in_dim = 100,
                            out_dim=8,
                            sim_multi=1,
                            ):
    data_v = x_normalizer.sample_generate(virtual_batchs*batch_size, 2, norm=False)

    post = cfdPost_2d()
    post.bouCondition_data_readin(
        boundarycondition=data_v[:, -4:],
    )

    field_matrix, bc_matrix = post.get_dimensional_matrix(expand=sim_multi, scale=scale)
    field_matrix = np.power(10, field_matrix)


    data_s = data_v.copy()
    data_s[:, -4:] = post.data_similarity_operate(data_v[:, -4:].copy(), bc_matrix)
    data_v = x_normalizer.norm(data_v)
    data_s = x_normalizer.norm(data_s)



    #
    # matrix_virtual = np.tile(matrix_virtual[:, None, None, :], [1, 64, 128, 1])
    # input_virtual = np.tile(input_virtual[:, None, None, :], [1, 64, 128, 1])
    data_s = torch.as_tensor(data_s, dtype=torch.float)
    data_v = torch.as_tensor(data_v, dtype=torch.float)

    virtual_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_v, field_matrix),
                                              batch_size=batch_size, shuffle=False, drop_last=True)
    similar_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_s, field_matrix),
                                                 batch_size=batch_size, shuffle=False, drop_last=True)
    return virtual_loader, similar_loader

def test_virtual_generation():
    in_dim = 100
    out_dim = 8


    batch_size = 32
    batch_number = 50

    ntrain = batch_number * batch_size
    nvalid = 500



    epochs = 1001
    learning_rate = 0.001
    scheduler_step = 700
    scheduler_gamma = 0.1

    design, fields, grids = get_origin(type='struct',
                                       realpath='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data')  # 获取原始数据取原始数据
    input = design
    input = np.tile(design[:, None, None, :], (1, 64, 128, 1))
    input = torch.tensor(input, dtype=torch.float)

    # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
    output = fields
    output = torch.tensor(output, dtype=torch.float)

    print(input.shape, output.shape)
    #
    train_x = input[:ntrain]
    train_y = output[:ntrain]
    # train_g = grids[:ntrain, ::r1]
    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]
    # valid_g = grids[-nvalid:, ::r1]
    #
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer.dim_change(2)

    x_normalizer_bc = DataNormer(train_x.numpy(), method='mean-std')
    x_normalizer_bc.dim_change(2)
    x_normalizer_bc.shrink(slice(96, 100, 1))

    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    y_normalizer.dim_change(2)

    self_loader = generate_virtual_loader(x_normalizer, batch_number, batch_size)




if __name__ == "__main__":
    test_virtual_generation()