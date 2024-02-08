import numpy as np
import yaml
import os
import torch
from Utilizes.process_data import DataNormer
from Tools.pre_process.data_readin import get_unstruct_quanlity_from_mat, get_struct_quanlity_from_mat
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
        sample_files = []
        basic = os.path.join(real_path, 'struct_'+str(hole_num)+'_hole_1000')
        if os.path.exists(basic + '.mat'):
            sample_files.append(basic)

        gather = os.path.join(real_path, 'gather_struct_'+str(hole_num)+'_hole_300')
        if os.path.exists(gather + '.mat'):
            sample_files.append(gather)

        test = os.path.join(real_path, 'struct_' + str(hole_num) + '_hole_120')
        if os.path.exists(test + '.mat'):
            sample_files.append(test)

    elif type == 'unstruct':
        sample_files = [os.path.join(real_path, 'unstruct_'+str(hole_num)+'_hole_1000'),
                        ]
    else:
        assert False

    return sample_files

def get_loader_pakB(
            train_x, train_y,
            x_normalizer=None,
            y_normalizer=None,
            batch_size = 32
           ):
    if x_normalizer is None:
        x_normalizer = DataNormer(train_x, method='mean-std', axis=(0,1,2,3))
    if y_normalizer is None:
        y_normalizer = DataNormer(train_y, method='mean-std')

    train_x = x_normalizer.norm(train_x)
    train_y = y_normalizer.norm(train_y)

    train_x = torch.as_tensor(train_x, dtype=torch.float)
    train_y = torch.as_tensor(train_y, dtype=torch.float)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, x_normalizer, y_normalizer


class PakBWeightLoss(torch.nn.Module):
    def __init__(self, weighted_cof=None, shreshold_cof=None, x_norm=None):
        super(PakBWeightLoss, self).__init__()

        if weighted_cof is None:
            weighted_cof = 0

        if shreshold_cof is None:
            shreshold_cof = 0

        self.lossfunc = torch.nn.MSELoss()
        self.weighted_cof = weighted_cof
        self.shreshold_cof = shreshold_cof
        self.x_norm = x_norm
    def forward(self, predicted, target, xx_mask):
        # 自定义损失计算逻辑
        device = target.device
        if xx_mask.shape[-1] > 1:
            xx_mask = xx_mask.min(dim=-1, keepdim=True).values
        xx_mask = self.x_norm.back(xx_mask)
        weight = (xx_mask > self.shreshold_cof).int()
        loss = self.lossfunc(predicted * weight, target * weight)
        return loss

class PakBAntiNormLoss(torch.nn.Module):
    def __init__(self, weighted_cof=None, shreshold_cof=None, x_norm=None, y_norm=None):
        super(PakBAntiNormLoss, self).__init__()
        if weighted_cof is None:
            weighted_cof = 0
        if shreshold_cof is None:
            shreshold_cof = 0

        self.lossfunc = torch.nn.MSELoss()
        self.weighted_cof = weighted_cof
        self.shreshold_cof = shreshold_cof
        self.x_norm = x_norm
        self.y_norm = y_norm
    def forward(self, predicted, target, xx_mask):
        # 自定义损失计算逻辑
        device = target.device
        if xx_mask.shape[-1] > 1:
            xx_mask = xx_mask.min(dim=-1, keepdim=True).values
        xx_mask = self.x_norm.back(xx_mask)
        weight = (xx_mask > self.shreshold_cof).int()
        loss = self.lossfunc(self.y_norm.back(predicted) * weight, self.y_norm.back(target) * weight)
        return loss

def clear_value_in_hole(pred, xx_mask, x_norm=None):
    if xx_mask.shape[-1] > 1:
        xx_mask = np.min(xx_mask, axis=-1, keepdims=True)
    xx_mask = x_norm.back(xx_mask)
    weight = (xx_mask > 0)
    pred[~weight] = np.nan
    return np.ma.masked_invalid(pred)

if __name__ == "__main__":
    os.chdir(r'E:\WQN\CODE\DENO4pytorch')
    design, field, grid = get_origin()
    print(0)