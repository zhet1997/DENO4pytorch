import numpy as np
import yaml
import os
import torch
from scipy.interpolate import interp1d
from Utilizes.data_readin import get_values_from_mat, get_unstruct_quanlity_from_mat, get_struct_quanlity_from_mat
def get_origin(quanlityList=None,
                type='struct',
                realpath=None,
                existcheck=True,
                shuffled=True,
                getridbad=True):

    if quanlityList is None:
        quanlityList = ["Static Pressure", "Static Temperature", "Density",
                        "Vx", "Vy", "Vz",
                        'Relative Total Temperature',
                        'Absolute Total Temperature',
                        ]

    sample_files = pakB_data_files(real_path=None)

    if existcheck:
        sample_files_exists = []
        for file in sample_files:
            if os.path.exists(file + '.mat'):
                sample_files_exists.append(file)
            else:
                print("The data file {} is not exist, CHECK PLEASE!".format(file))

        sample_files = sample_files_exists


    if type == 'struct':
        grid, fields, invalid_idx = get_struct_quanlity_from_mat(sample_files, quanlityList=quanlityList,
                                                                   invalid_idx=True)
    elif type == 'unstruct':
        grid, fields, invalid_idx = get_unstruct_quanlity_from_mat(sample_files, quanlityList=quanlityList, invalid_idx=True)

    dict = get_values_from_mat(geom_files, keyList=['design', 'gemo', 'condition'])
    design = np.concatenate((dict['design'],dict['condition']), axis=-1)

    if getridbad:
        if realpath is None:
            file_path = os.path.join("../Rotor37_2d/data", "sus_bad_data.yml")
        else:
            file_path = os.path.join(realpath, "sus_bad_data.yml")

        sus_bad_idx = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                sus_bad_dict = yaml.load(f, Loader=yaml.FullLoader)
            for key in sus_bad_dict.keys():
                sus_bad_idx.extend(sus_bad_dict[key])
            sus_bad_idx = np.array(sus_bad_idx)
        sus_bad_idx = np.unique(np.concatenate((sus_bad_idx,np.array(invalid_idx)),axis=0))

        design = np.delete(design, sus_bad_idx.astype('int64'), axis=0)
        fields = np.delete(fields, sus_bad_idx.astype('int64'), axis=0)
        if type == 'unstruct':
            grid = np.delete(grid, sus_bad_idx.astype('int64'), axis=0)

    if shuffled:
        np.random.seed(8905)
        idx = np.random.permutation(design.shape[0])
        design = design[idx]
        fields = fields[idx]
        if type == 'unstruct':
            grid = grid[idx]

    return design, fields, grid

def pakB_data_files(real_path=None):
    if type == 'struct':
        sample_files = [os.path.join(real_path, "sampleStruct_128_64_6000"),
                        ]
    elif type == 'unstruct':
        sample_files = [os.path.join(real_path, 'GV-RB3000(20231015)',"sampleUnstruct_3000"),
                        os.path.join(real_path, 'GV-RB3000(20231017)',"sampleUnstruct_3000"),
                        ]
    else:
        assert False

    return sample_files


if __name__ == "__main__":
    design, field = get_origin()
    # grid = get_grid()
    # Rst = get_value(field, parameterList="PressureRatioW")
    #
    # sort_idx = np.argsort(Rst.squeeze())
    # sort_value = Rst[sort_idx]
    # design = get_gemodata_from_mat()
    # print(design.shape)


    print(0)