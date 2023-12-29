import numpy as np
import yaml
import torch
from Utilizes.process_data import DataNormer, MatLoader
from Tools.post_process.post_data import Post_2d
import os
import torch
from scipy.interpolate import interp1d

def get_grid(real_path=None):
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    if real_path is None:
        hub_file = os.path.join('../Rotor37_2d/data', 'hub_lower.txt')
        shroud_files = os.path.join('../Rotor37_2d/data', 'shroud_upper.txt')
    else:
        hub_file = os.path.join(real_path, 'hub_lower.txt')
        shroud_files = os.path.join(real_path, 'shroud_upper.txt')

    hub = np.loadtxt(hub_file)
    shroud = np.loadtxt(shroud_files)

    yy = []
    for i in range(64):
        yy.append(np.linspace(hub[i],shroud[i],64))

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(64, 64).T
    xx = xx.reshape(64, 64)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)

def get_grid_interp(grid_num_s=128,
                    grid_num_z=128,
                    z_inlet=-0.059,
                    z_outlet=0.119,
                    hub_adjust=0.00015,
                    shroud_adjust=0.0001,
                    ):
    shroud_file = os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data', "shroud.dat")
    hub_file = os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data', "hub.dat")
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
    if realpath is None:
        if type=='struct':
            sample_files = [os.path.join("data", "sampleStruct_128_64_6000"),
                            ]
        elif type=='unstruct':
            sample_files = [os.path.join("data", 'GV-RB3000(20231015)', "sampleUnstruct_3000"),
                            os.path.join("data", 'GV-RB3000(20231017)', "sampleUnstruct_3000"),
                            ]
        else:
            assert False
        geom_files =   [os.path.join("data", 'GV-RB3000(20231015)', "sampleGVM_3000"),
                        os.path.join("data", 'GV-RB3000(20231017)', "sampleGVM_3000"),
                        ]
    else:
        if type == 'struct':
            sample_files = [os.path.join(realpath, "sampleStruct_128_64_6000"),
                            ]
        elif type == 'unstruct':
            sample_files = [os.path.join(realpath, 'GV-RB3000(20231015)',"sampleUnstruct_3000"),
                            os.path.join(realpath, 'GV-RB3000(20231017)',"sampleUnstruct_3000"),
                            ]
        else:
            assert False
        geom_files = [os.path.join(realpath, 'GV-RB3000(20231015)', "sampleGVM_3000"),
                      os.path.join(realpath, 'GV-RB3000(20231017)', "sampleGVM_3000"),
                      ]

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


def get_struct_quanlity_from_mat(sample_files, quanlityList,invalid_idx=True):
    grid = []
    fields = []
    invalid_idx_list = []
    data_sum = 0
    if not isinstance(sample_files, list):
        sample_files = [sample_files]
    for ii, file in enumerate(sample_files):
        reader = MatLoader(file, to_torch=False)
        grid.append(reader.read_field('grid'))
        temp = [x for x in reader.data.keys() if not x.startswith('__') and not x == 'grid' and not x == 'invalid_idx']
        data_shape = reader.read_field(temp[0]).shape
        output = np.zeros([*data_shape,len(quanlityList)])
        if invalid_idx:
            idx = reader.read_field('invalid_idx')
            idx = [int(x+data_sum) for x in idx.squeeze()]
            invalid_idx_list.extend(idx)
        Cp = 1004
        for jj, quanlity in enumerate(quanlityList):
            if quanlity == "DensityFlow":  # 设置一个需要计算获得的数据
                Vm = np.sqrt(np.power(reader.read_field("Vxyz_X"), 2) + np.power(reader.read_field("Vxyz_Y"), 2))
                output[:, :, :, jj] = (reader.read_field("Density") * Vm).copy()
            elif quanlity == "W2":  # 设置一个需要计算获得的数据
                output[:, :, :, jj] = 2 * Cp * (reader.read_field("Relative Total Temperature") - reader.read_field(
                    "Static Temperature")).copy()
            elif quanlity == "V2":  # 设置一个需要计算获得的数据
                output[:, :, :, jj] = 2 * Cp * (reader.read_field("Absolute Total Temperature") - reader.read_field(
                    "Static Temperature")).copy()
            else:
                output[:, :, :, jj] = reader.read_field(quanlity).copy()
        fields.append(output)

    grid = np.concatenate(grid, axis=0)
    fields = np.concatenate(fields, axis=0)
    if invalid_idx:
        return grid, fields, invalid_idx_list
    else:
        return grid, fields

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

class Rotor37WeightLoss(torch.nn.Module):
    def __init__(self):
        super(Rotor37WeightLoss, self).__init__()

        self.lossfunc = torch.nn.MSELoss()

        self.weighted_lines = 2
        self.weighted_cof = 10

    def forward(self, predicted, target):
        # 自定义损失计算逻辑
        device = target.device
        if target.shape[1] > 4000:
            target = torch.reshape(target, (target.shape[0], 64, 64, -1))
            predicted = torch.reshape(predicted, (target.shape[0], 64, 64, -1))

        if len(target.shape)==3:
            predicted = predicted.unsqueeze(0)
        if len(target.shape)==2:
            predicted = predicted.unsqueeze(0).unsqueeze(-1) #加一个维度

        grid_size_1 = target.shape[1]
        grid_size_2 = target.shape[2]

        temp1 = torch.ones((grid_size_1, self.weighted_lines), dtype=torch.float32, device=device) * self.weighted_cof
        temp2 = torch.ones((grid_size_1, grid_size_2 - self.weighted_lines * 2), dtype=torch.float32, device=device)
        weighted_mat = torch.cat((temp1, temp2, temp1), dim=1)
        weighted_mat = weighted_mat.unsqueeze(0).unsqueeze(-1).expand_as(target)
        weighted_mat = weighted_mat * grid_size_2 /(self.weighted_cof * self.weighted_lines * 2
                                                    + grid_size_2 - self.weighted_lines * 2)

        loss = self.lossfunc(predicted * weighted_mat, target * weighted_mat)
        return loss

class GVRBWeightLoss(torch.nn.Module):
    def __init__(self, weighted_lines, weighted_cof, RS_position):
        super(GVRBWeightLoss, self).__init__()

        self.lossfunc = torch.nn.MSELoss()

        self.weighted_lines = weighted_lines
        self.weighted_cof = weighted_cof
        self.RS_position = RS_position

    def forward(self, predicted, target):
        # 自定义损失计算逻辑
        device = target.device
        if target.shape[1] > 4000:
            target = torch.reshape(target, (target.shape[0], 64, 128, -1))
            predicted = torch.reshape(predicted, (target.shape[0], 64, 128, -1))

        if len(target.shape)==3:
            predicted = predicted.unsqueeze(0)
        if len(target.shape)==2:
            predicted = predicted.unsqueeze(0).unsqueeze(-1) #加一个维度

        grid_size_1 = target.shape[1]
        grid_size_2 = target.shape[2]

        temp_LT = torch.ones((grid_size_1, self.weighted_lines), dtype=torch.float32, device=device) * self.weighted_cof
        temp_RS = torch.ones((grid_size_1, self.weighted_lines+2), dtype=torch.float32, device=device) * self.weighted_cof
        temp_S = torch.ones((grid_size_1, int(self.RS_position-self.weighted_lines-(self.weighted_lines+2)/2)), dtype=torch.float32, device=device)
        temp_R = torch.ones((grid_size_1, int(grid_size_2-self.RS_position-self.weighted_lines-(self.weighted_lines+2)/2)), dtype=torch.float32,
                            device=device)
        weighted_mat = torch.cat((temp_LT, temp_S, temp_RS, temp_R, temp_LT), dim=1)
        weighted_mat = weighted_mat.unsqueeze(0).unsqueeze(-1).expand_as(target)
        weighted_mat = weighted_mat * grid_size_2 /(self.weighted_cof * (self.weighted_lines *3 + 2)
                                                    + grid_size_2 - (self.weighted_lines *3 + 2))

        loss = self.lossfunc(predicted * weighted_mat, target * weighted_mat)
        return loss

class SelfSuperviseLoss(torch.nn.Module):
    def __init__(self,):
        super(SelfSuperviseLoss, self).__init__()

        self.lossfunc = torch.nn.MSELoss()
    def forward(self, predicted, field_matrix, y_norm=None):
        # 自定义损失计算逻辑
        device = predicted.device

        tmp = y_norm.back(predicted)
        tmp = tmp/field_matrix
        predicted = y_norm.norm(tmp)

        var = torch.var(predicted, dim=0)
        zeros = torch.zeros_like(var)

        loss = self.lossfunc(var, zeros)
        return loss

class SelfSuperviseLoss2(torch.nn.Module):
    def __init__(self, ):
        super(SelfSuperviseLoss2, self).__init__()

        self.lossfunc = torch.nn.MSELoss()

    def forward(self, predicted, field_matrix, y_norm=None):
        # 自定义损失计算逻辑
        device = predicted.device

        tmp = y_norm.back(predicted)
        tmp = tmp / field_matrix
        predicted = y_norm.norm(tmp)

        target = predicted[0:1,...].repeat(predicted.shape[0], 1, 1, 1)

        loss = self.lossfunc(predicted, target)
        return loss

class SelfSuperviseLoss3(torch.nn.Module):
    def __init__(self, ):
        super(SelfSuperviseLoss3, self).__init__()

        self.lossfunc = torch.nn.MSELoss()

    def forward(self, predicted, field_matrix, y_norm=None):
        # 自定义损失计算逻辑
        device = predicted.device
        num = int(predicted.shape[0]/2)

        tmp = y_norm.back(predicted)
        tmp = tmp / field_matrix
        predicted = y_norm.norm(tmp)

        # target = predicted[0:1,...].repeat(predicted.shape[0], 1, 1, 1)

        loss = self.lossfunc(predicted[:num,...].detach(), predicted[num:,...])
        return loss

class SelfSuperviseLoss4(torch.nn.Module):# variance
    def __init__(self, ):
        super(SelfSuperviseLoss4, self).__init__()

        self.lossfunc = torch.nn.MSELoss()

    def forward(self, predicted, field_matrix, y_norm=None):
        # 自定义损失计算逻辑
        device = predicted.device
        num = int(predicted.shape[0]/2)

        tmp = y_norm.back(predicted[:num,...])
        std = torch.sqrt(tmp.var(dim=(1,2), keepdim=True) + 1e-5)
        mean = tmp.mean(dim=(1,2), keepdim=True)
        predicted_norm = (tmp-mean)/std #已经归一化完了，不用再norm了。


        var = torch.sqrt(predicted_norm.var(dim=0) + 1e-5)
        delta = 0.07
        std_loss = torch.mean(torch.nn.functional.relu(delta-var))


        # target = predicted[0:1,...].repeat(predicted.shape[0], 1, 1, 1)

        loss = self.lossfunc(predicted[:num,...], predicted[num:,...])
        return std_loss

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

    # np.savetxt(os.path.join("Rst.txt"), Rst)
    # file_path = os.path.join("data", "sus_bad_data.yml")
    # import yaml
    # with open(file_path,'r') as f:
    #     data = yaml.load(f, Loader=yaml.FullLoader)