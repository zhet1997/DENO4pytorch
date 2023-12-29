import numpy as np
import yaml
import os
import torch
from Utilizes.data_readin import get_values_from_mat, get_unstruct_quanlity_from_mat, get_struct_quanlity_from_mat
def get_origin(quanlityList=None,
                type='struct',
                hole_num=1,
                realpath=None,
                existcheck=True,
                shuffled=True,
                getridbad=True
               ):

    if quanlityList is None:
        quanlityList_i = ["sdf",
                        ]
        quanlityList_o = ["Temperature",
                          ]
    if realpath is None:
        realpath = os.path.join('Demo', 'PakB_2d', 'data')
    sample_files = pakB_data_files(real_path=realpath, type=type, hole_num=hole_num)

    if existcheck:
        sample_files_exists = []
        for file in sample_files:
            if os.path.exists(file + '.mat'):
                sample_files_exists.append(file)
            else:
                print("The data file {} is not exist, CHECK PLEASE!".format(file))

        sample_files = sample_files_exists


    if type == 'struct':
        design, grid, fields, invalid_idx = get_struct_quanlity_from_mat(sample_files,
                                                                 quanlityList_i=quanlityList_i,
                                                                 quanlityList_o=quanlityList_o,
                                                                   invalid_idx=True)
    elif type == 'unstruct':
        grid, fields, invalid_idx = get_unstruct_quanlity_from_mat(sample_files, quanlityList=quanlityList, invalid_idx=True)

    # dict = get_values_from_mat(geom_files, keyList=['design', 'gemo', 'condition'])
    # design = dict['sdf']

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

def pakB_data_files(real_path=None, type=None, hole_num=1):
    if type == 'struct':
        sample_files = [os.path.join(real_path, 'struct_'+str(hole_num)+'_hole_1000'),
                        ]
    elif type == 'unstruct':
        sample_files = [os.path.join(real_path, 'unstruct_'+str(hole_num)+'_hole_1000'),
                        ]
    else:
        assert False

    return sample_files


class PakBWeightLoss(torch.nn.Module):
    def __init__(self, weighted_cof):
        super(PakBWeightLoss, self).__init__()

        self.lossfunc = torch.nn.MSELoss()
        self.weighted_cof = weighted_cof
    def forward(self, ww, predicted, target):
        # 自定义损失计算逻辑
        device = target.device
        # xx, _ = torch.min(xx, dim=0)
        weight = torch.as_tensor(ww, dtype=torch.float , device=device)#(xx > 0).int()
        loss = self.lossfunc(predicted * weight[...,None], target * weight[...,None])
        return loss


if __name__ == "__main__":
    os.chdir(r'E:\WQN\CODE\DENO4pytorch')
    design, field, grid = get_origin()
    print(0)