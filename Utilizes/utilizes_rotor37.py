import numpy as np
import yaml
import paddle
from Utilizes.process_data import MatLoader
from post_process.post_data import Post_2d
import os

def get_grid(real_path=None):
    xx = np.linspace(-0.127, 0.126, 64)
    xx = np.tile(xx, [64,1])

    if real_path is None:
        hub_file = os.path.join('../Demo/Rotor37_2d/data', 'hub_lower.txt')
        shroud_files = os.path.join('../Demo/Rotor37_2d/data', 'shroud_upper.txt')
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

def get_origin(quanlityList=None,
                realpath=None,
                existcheck=True,
                shuffled=True,
                getridbad=True):

    if quanlityList is None:
        quanlityList = ["Static Pressure", "Static Temperature",
                        'V2', 'W2', "DensityFlow"]
    if realpath is None:
        sample_files = [os.path.join("../Demo/Rotor37_2d/data", "sampleRstZip_1500"),
                        os.path.join("../Demo/Rotor37_2d/data", "sampleRstZip_500"),
                        os.path.join("../Demo/Rotor37_2d/data", "sampleRstZip_970")
                        ]
    else:
        sample_files = [os.path.join(realpath, "sampleRstZip_1500"),
                        os.path.join(realpath, "sampleRstZip_500"),
                        os.path.join(realpath, "sampleRstZip_970")
                        ]
    if existcheck:
        sample_files_exists = []
        for file in sample_files:
            if os.path.exists(file + '.mat'):
                sample_files_exists.append(file)
            else:
                print("The data file {} is not exist, CHECK PLEASE!".format(file))

        sample_files = sample_files_exists

    design, fields = get_quanlity_from_mat(sample_files, quanlityList)

    if getridbad:
        if realpath is None:
            file_path = os.path.join("../Demo/Rotor37_2d/data", "sus_bad_data.yml")
        else:
            file_path = os.path.join(realpath, "sus_bad_data.yml")
        with open(file_path, 'r') as f:
            sus_bad_dict = yaml.load(f, Loader=yaml.FullLoader)
        sus_bad_idx = []
        for key in sus_bad_dict.keys():
            sus_bad_idx.extend(sus_bad_dict[key])
        sus_bad_idx = np.array(sus_bad_idx)
        sus_bad_idx = np.unique(sus_bad_idx)

        design = np.delete(design, sus_bad_idx, axis=0)
        fields = np.delete(fields, sus_bad_idx, axis=0)

    if shuffled:
        np.random.seed(8905)
        idx = np.random.permutation(design.shape[0])
        # print(idx[:10])
        design = design[idx]
        fields = fields[idx]

    return design, fields

def get_quanlity_from_mat(sample_files, quanlityList):
    design = []
    fields = []
    if not isinstance(sample_files, list):
        sample_files = [sample_files]
    for ii, file in enumerate(sample_files):
        reader = MatLoader(file, to_torch=False)
        design.append(reader.read_field('design'))
        output = np.zeros([design[ii].shape[0], 64, 64, len(quanlityList)])
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

    design = np.concatenate(design, axis=0)
    fields = np.concatenate(fields, axis=0)

    return design, fields



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

class Rotor37WeightLoss(paddle.nn.Layer):
    def __init__(self):
        super(Rotor37WeightLoss, self).__init__()

    def forward(self, predicted, target):
        # 自定义损失计算逻辑
        # device = target.device
        if target.shape[1] > 4000:
            target = paddle.reshape(target, (target.shape[0], 64, 64, -1))
            predicted = paddle.reshape(predicted, (target.shape[0], 64, 64, -1))

        if len(target.shape)==3:
            predicted = predicted.unsqueeze(0)
        if len(target.shape)==2:
            predicted = predicted.unsqueeze(0).unsqueeze(-1) #加一个维度

        grid_size_1 = target.shape[1]
        grid_size_2 = target.shape[2]
        weighted_lines = 2
        weighted_cof = 10

        temp1 = paddle.ones((grid_size_1, weighted_lines)) * weighted_cof
        temp2 = paddle.ones((grid_size_1, grid_size_2 - weighted_lines * 2))
        weighted_mat = paddle.concat((temp1, temp2, temp1), axis=1)
        weighted_mat = weighted_mat.unsqueeze(0).unsqueeze(-1).expand_as(target)
        weighted_mat = weighted_mat * grid_size_2 /(weighted_cof * weighted_lines * 2 + grid_size_2 - weighted_lines * 2)
        weighted_mat = weighted_mat
        lossfunc = paddle.nn.MSELoss()
        loss = lossfunc(predicted * weighted_mat, target * weighted_mat)
        return loss

