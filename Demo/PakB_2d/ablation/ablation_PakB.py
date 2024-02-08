import os
import numpy as np
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBWeightLoss, get_loader_pakB, clear_value_in_hole
from Tools.pre_process.data_reform import data_padding, split_train_valid, get_loader_from_list, get_loader_from_unpadding_list
import yaml
def get_loaders(dataset_train,
                dataset_valid,
                train_num=500,
                valid_num=200,
                channel_num=10,
                batch_size=32,
                channel_shuffle=False,
                batch_shuffle=True,
                unpadding=True
                ):
    train_input_list = []
    train_output_list = []
    valid_input_list = []
    valid_output_list = []
    train_input_dict = {}
    train_output_dict = {}
    valid_input_dict = {}
    valid_output_dict = {}
    dataset = np.unique(np.array(dataset_train + dataset_valid)).tolist()
    # load all data
    for kk, hole_num in enumerate(dataset):
        design, fields, grids = get_origin(type='struct', hole_num=hole_num, realpath=os.path.join('data'))
        print(design.shape)# 获取原始数据取原始数据
        # input = data_padding(design, const=350, channel_num=channel_num, shuffle=channel_shuffle)
        input = design
        output = fields

        train_i, valid_i = split_train_valid(input, train_num=train_num, valid_num=valid_num)
        train_o, valid_o = split_train_valid(output, train_num=train_num, valid_num=valid_num)
        print(train_i.shape, train_o.shape)
        print(valid_i.shape, valid_o.shape)

        train_input_dict.update({str(hole_num): train_i.copy()})
        train_output_dict.update({str(hole_num): train_o.copy()})
        valid_input_dict.update({str(hole_num): valid_i.copy()})
        valid_output_dict.update({str(hole_num): valid_o.copy()})

    for hole_num in dataset_train:
        train_input_list.append(train_input_dict[str(hole_num)])
        train_output_list.append(train_output_dict[str(hole_num)])

    for hole_num in dataset_valid:
        valid_input_list.append(valid_input_dict[str(hole_num)])
        valid_output_list.append(valid_output_dict[str(hole_num)])

    if unpadding:
        combine_func = get_loader_from_unpadding_list
    else:
        combine_func = get_loader_from_list

    train_loader, x_normalizer, y_normalizer = combine_func(train_input_list,
                                                                   train_output_list,
                                                                   batch_size=batch_size,
                                                                   shuffle=batch_shuffle,
                                                                   combine_list=True,
                                                                   )
    valid_loader_list, _, _ = combine_func(valid_input_list,
                                                   valid_output_list,
                                                   x_normalizer=x_normalizer,
                                                   y_normalizer=y_normalizer,
                                                   batch_size=batch_size,
                                                   shuffle=batch_shuffle,
                                                   combine_list=False,
                                                   padding=False,
                                                   )
    return train_loader, valid_loader_list, x_normalizer, y_normalizer
    # ################################################################


def get_setting():
    basic_dict = {
        'in_dim': 10,
        'out_dim': 1,
        'ntrain': 500,
        'nvalid': 100,
    }
    train_dict = {
        'batch_size': 32,
        'epochs': 801,
        'learning_rate': 0.001,
        'scheduler_step': 700,
        'scheduler_gamma': 0.1,
    }
    super_model_dict = {
        'modes': (16, 16),
        'width': 64,
        'depth': 2,
        'steps': 1,
        'padding': 0,
        'dropout': 0.1,
    }
    with open(os.path.join('data', 'configs', 'transformer_config_pakb.yml')) as f:
        config = yaml.full_load(f)
        pred_model_dict = config['PakB_2d']
        pred_model_dict['node_feats'] = basic_dict['in_dim']

    return basic_dict, train_dict, pred_model_dict, super_model_dict


# def calculate_per_0(n):
#     permutations_list = []
#     total_sum = np.math.factorial(n)
#     for k in range(n + 1):
#         permutation = np.math.perm(n, k)
#         permutations_list.append(float(permutation/sum))

    return permutations_list

def calculate_per(n):
    permutations_list = []
    for k in range(n + 1):
        permutation = np.math.factorial(n)/np.math.factorial(n-k)/np.math.factorial(k)/ np.power(2, n)
        permutations_list.append(permutation)

    return permutations_list