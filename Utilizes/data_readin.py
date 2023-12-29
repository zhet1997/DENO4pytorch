import numpy as np
import yaml
import torch
from Utilizes.process_data import DataNormer, MatLoader
from Tools.post_process.post_data import Post_2d
import os
import torch
from scipy.interpolate import interp1d

def get_grid_interp(grid_num_s=128,
                    grid_num_z=128,
                    z_inlet=-0.059,
                    z_outlet=0.119,
                    hub_adjust=0.00015,
                    shroud_adjust=0.0001,
                    ):
    shroud_file = os.path.join('..','data',"shroud.dat")
    hub_file = os.path.join('..','data',"hub.dat")
    hub = np.loadtxt(hub_file)/1000
    shroud = np.loadtxt(shroud_file)/1000

    x = np.linspace(z_inlet, z_outlet, grid_num_z)
    xx = np.tile(x, [grid_num_s, 1])

    f_hub = interp1d(hub[:, 0], hub[:, 1], kind='linear')
    y_hub = f_hub(x)
    f_shroud = interp1d(shroud[:, 0], shroud[:, 1], kind='linear')
    y_shroud = f_shroud(x)

    yy = []
    for i in range(grid_num_z):
        yy.append(np.linspace(y_hub[i]+hub_adjust,y_shroud[i]-shroud_adjust,grid_num_s)) # check

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(grid_num_z, grid_num_s).T
    xx = xx.reshape(grid_num_s, grid_num_z)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)


def get_unstruct_quanlity_from_mat(sample_files, quanlityList,invalid_idx=True):
    grid = []
    fields = []
    invalid_idx_list = []
    data_sum = 0
    if not isinstance(sample_files, list):
        sample_files = [sample_files]
    for ii, file in enumerate(sample_files):
        reader = MatLoader(file, to_torch=False)
        grid.append(reader.read_field('grid'))
        temp = [x for x in reader.data.keys() if not x.startswith('__')]
        data_shape = reader.read_field(temp[0]).shape
        output = np.zeros([data_shape[0], data_shape[1], len(quanlityList)])
        if invalid_idx:
            idx = reader.read_field('invalid_idx')
            idx = [int(x+data_sum) for x in idx.squeeze()]
            invalid_idx_list.extend(idx)
        Cp = 1004
        data_sum =+ data_shape[0]
        for jj, quanlity in enumerate(quanlityList):
            if quanlity == "DensityFlow":  # 设置一个需要计算获得的数据
                Vm = np.sqrt(np.power(reader.read_field("Vxyz_X"), 2) + np.power(reader.read_field("Vxyz_Y"), 2))
                output[..., jj] = (reader.read_field("Density") * Vm).copy()
            elif quanlity == "W2":  # 设置一个需要计算获得的数据
                output[..., jj] = 2 * Cp * (reader.read_field("Relative Total Temperature") - reader.read_field(
                    "Static Temperature")).copy()
            elif quanlity == "V2":  # 设置一个需要计算获得的数据
                output[..., jj] = 2 * Cp * (reader.read_field("Absolute Total Temperature") - reader.read_field(
                    "Static Temperature")).copy()
            else:
                output[..., jj] = reader.read_field(quanlity).copy()
        fields.append(output)

    grid = np.concatenate(grid, axis=0)
    fields = np.concatenate(fields, axis=0)
    if invalid_idx:
        return grid, fields, invalid_idx_list
    else:
        return grid, fields


def get_struct_quanlity_from_mat(sample_files, quanlityList_i, quanlityList_o,invalid_idx=True):
    invalid_idx_list = []
    output_list = []
    data_sum = 0
    if not isinstance(sample_files, list):
        sample_files = [sample_files]
    for ii, file in enumerate(sample_files):
        reader = MatLoader(file, to_torch=False)
        temp = [x for x in reader.data.keys() if not x.startswith('__') and not x == 'invalid_idx']

        data_shape = reader.read_field(temp[0]).shape
        output = {}
        if invalid_idx:
            idx = reader.read_field('invalid_idx')
            idx = [int(x+data_sum) for x in idx.squeeze()]
            invalid_idx_list.extend(idx)

        for quanlity in quanlityList_i + quanlityList_o:
           assert quanlity in temp

        for jj, quanlity in enumerate(quanlityList_i + quanlityList_o +['Grids_x', 'Grids_y']):
            output.update({quanlity: reader.read_field(quanlity).copy()})
        output_list.append(output)

    designs = [np.concatenate(list(x[quanlity] for quanlity in quanlityList_i), axis=-1) for x in output_list]
    designs = np.concatenate(designs, axis=0)

    fields = [np.concatenate(list(x[quanlity][...,None] for quanlity in quanlityList_o), axis=-1) for x in output_list]
    fields = np.concatenate(fields, axis=0)

    if 'grid' in temp:
        grid = [x['grid'] for x in output_list]
    elif 'Grids_x' in temp:
        grid = [np.concatenate((x['Grids_x'][...,None], x['Grids_y'][...,None]), axis=-1) for x in output_list]
    else:
        assert False
    grid = np.concatenate(grid, axis=0)


    if invalid_idx:
        return designs, grid, fields, invalid_idx_list
    else:
        return designs, grid, fields

def get_values_from_mat(sample_files, keyList=None):
    dict = {}
    if not isinstance(sample_files, list):
        sample_files = [sample_files]
    if not isinstance(keyList, list):
        assert keyList is not None
        keyList = [keyList]
    for jj, key in enumerate(keyList):
        dict.update({key: []})
        for ii, file in enumerate(sample_files):
            temp = []
            reader = MatLoader(file, to_torch=False)
            output = reader.read_field(key).copy()
            dict[key].append(output)
        dict[key] = np.concatenate(dict[key], axis=0)

    return dict

def get_value(data_2d, input_para=None, parameterList=None):
    if not isinstance(parameterList, list):
        parameterList = [parameterList]

    grid = get_grid()
    post_pred = Post_2d(data_2d, grid,
                        inputDict=input_para,
                        )

    Rst = []
    for parameter_Name in parameterList:
        value = getattr(post_pred, parameter_Name)
        value = post_pred.span_density_average(value[..., -1])
        Rst.append(value)

    return np.concatenate(Rst, axis=1)
