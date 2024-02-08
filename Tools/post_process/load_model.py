import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin,get_origin_GVRB
from Utilizes.process_data import DataNormer
import yaml
# from utilizes_rotor37 import get_origin_GVRB
from Demo.GVRB_2d.utilizes_GVRB import get_grid, get_origin
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid1



def get_noise(shape, scale):
    random_array = np.random.randn(np.prod(shape)) #  randn生成的是标准正态分布
    # random_array = (random_array-1)*2
    random_array = random_array.reshape(shape)

    return random_array * scale

def loaddata_Sql(name,
            ntrain=3000,
            nvalid=900,
            shuffled=False,
            noise_scale=None,
            batch_size=32,
            norm_x=None,
            norm_y=None,
            norm_method='mean-std',
                 ):
    design, fields, grids = get_origin(type='struct', realpath='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data/',
                                       quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                     "Vx", "Vy", "Vz",
                                                     'Relative Total Temperature',
                                                     'Absolute Total Temperature',
                                                     ],
                                                                              )
    nameReal = name.split("_")[0]
    id = None
    if len(name.split("_")) == 2:
        id = int(name.split("_")[1])

    name = nameReal
    if name in ("FNO", "FNM", "UNet", "Transformer"):
        input = np.tile(design[:, None, None, :], (1, 64, 128, 1))

    else:
        input = design

    output = fields

    input = torch.tensor(input, dtype=torch.float)
    output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)


    train_x = input[:ntrain]
    train_y = output[:ntrain]


    valid_x = input[-nvalid:]
    valid_y = output[-nvalid:]

    if norm_x is None:
        x_normalizer = DataNormer(train_x, method=norm_method)
    else:
        x_normalizer = norm_x
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    if norm_y is None:
        y_normalizer = DataNormer(train_y, method=norm_method)
    else:
        y_normalizer = norm_y
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    if name in ("MLP"):
        train_y = train_y.reshape([train_y.shape[0], -1])
        valid_y = valid_y.reshape([valid_y.shape[0], -1])

    if noise_scale is not None and noise_scale > 0: # 向数据中增加噪声
        noise_train = get_noise(train_y.shape, noise_scale)
        train_y = train_y + noise_train

    # 完成了归一化后再转换数据
    train_x = torch.as_tensor(train_x, dtype=torch.float)
    train_y = torch.as_tensor(train_y, dtype=torch.float)
    valid_x = torch.as_tensor(valid_x, dtype=torch.float)
    valid_y = torch.as_tensor(valid_y, dtype=torch.float)

    if name in ("deepONet"):
        # grid = get_grid(real_path=os.path.join("../../Demo/Rotor37_2d", "data"))
        design, fields, grids = get_origin(type='struct', realpath='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data/',
                                           quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                         "Vx", "Vy", "Vz",
                                                         'Relative Total Temperature',
                                                         'Absolute Total Temperature'])

        input = np.tile(design[:, None, None, :], (1, 64, 128, 1))
        input = torch.tensor(input, dtype=torch.float)

        # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
        output = fields
        output = torch.tensor(output, dtype=torch.float)
        grids = np.tile(grids[None, ...], (design.shape[0], 1, 1, 1))
        grids = torch.tensor(grids, dtype=torch.float)
        print(input.shape, output.shape)
        r1 = 1
        train_x = input[:ntrain, ::r1]
        train_y = output[:ntrain, ::r1]
        train_g = grids[:ntrain, ::r1]
        valid_x = input[-nvalid:, ::r1]
        valid_y = output[-nvalid:, ::r1]
        valid_g = grids[-nvalid:, ::r1]

        x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
        train_x = x_normalizer.norm(train_x)
        valid_x = x_normalizer.norm(valid_x)

        y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
        train_y = y_normalizer.norm(train_y)
        valid_y = y_normalizer.norm(valid_y)

        g_normalizer = DataNormer(train_g.numpy(), method='mean-std')
        train_g = g_normalizer.norm(train_g)
        valid_g = g_normalizer.norm(valid_g)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_g, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_g, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader, x_normalizer, y_normalizer

def loaddata(name,
             ntrain=1500,
             nvalid=500,
             shuffled=False,
             noise_scale=None,
             batch_size=32):

    design, fields = get_origin(realpath=os.path.join("../../Demo/TwoLPT_2d", "data"), shuffled=shuffled)  # 获取原始数据

    if name in ("FNO", "FNM", "UNet"):
        input = np.tile(design[:, None, None, :], (1, 64, 64, 1))
    else:
        input = design
    output = fields

    # input = torch.tensor(input, dtype=torch.float)
    # output = torch.tensor(output, dtype=torch.float)
    print(input.shape, output.shape)

    train_x = input[:ntrain, :]
    train_y = output[:ntrain, :]
    valid_x = input[-nvalid:, :]
    valid_y = output[-nvalid:, :]

    x_normalizer = DataNormer(train_x, method='mean-std')
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)

    y_normalizer = DataNormer(train_y, method='mean-std')
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)

    if name in ("MLP"):
        train_y = train_y.reshape([train_y.shape[0], -1])
        valid_y = valid_y.reshape([valid_y.shape[0], -1])

    if noise_scale is not None and noise_scale > 0: # 向数据中增加噪声
        noise_train = get_noise(train_y.shape, noise_scale)
        train_y = train_y + noise_train

    # 完成了归一化后再转换数据
    train_x = torch.tensor(train_x, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.float)
    valid_x = torch.tensor(valid_x, dtype=torch.float)
    valid_y = torch.tensor(valid_y, dtype=torch.float)

    if name in ("deepONet"):
        grid = get_grid(real_path=os.path.join("../../Demo/Rotor37_2d", "data"))
        grid_trans = torch.tensor(grid[np.newaxis, :, :, :], dtype=torch.float)
        train_grid = torch.tile(grid_trans, [train_x.shape[0], 1, 1, 1])  # 所有样本的坐标是一致的。
        valid_grid = torch.tile(grid_trans, [valid_x.shape[0], 1, 1, 1])

        grid_normalizer = DataNormer(train_grid.numpy(), method='mean-std')  # 这里的axis不一样了
        train_grid = grid_normalizer.norm(train_grid)
        valid_grid = grid_normalizer.norm(valid_grid)

        # grid_trans = grid_trans.reshape([1, -1, 2])
        train_grid = train_grid.reshape([train_x.shape[0], -1, 2])
        valid_grid = valid_grid.reshape([valid_x.shape[0], -1, 2])
        train_y = train_y.reshape([train_y.shape[0], -1, 5])
        valid_y = valid_y.reshape([valid_y.shape[0], -1, 5])

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_grid, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_grid, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader, x_normalizer, y_normalizer

def rebuild_model(work_path, Device, in_dim=92, out_dim=4, name=None, mode=10):
    """
    rebuild the model with pth files
    """
    # rebuild the model
    if 'MLP' in name:
        from run_MLP import MLP
        from run_MLP import inference
        layer_mat = [in_dim, 256, 256, 256, 256, 256, 256, 256, 256, out_dim * 64 * 64]
        Net_model = MLP(layer_mat, is_BatchNorm=False).to(Device)
    elif 'deepONet' in name:
        from don.DeepONets import DeepONetMulti
        from run_deepONet import inference
        Net_model = DeepONetMulti(input_dim=2, operator_dims=[28, ], output_dim=5,
                                  planes_branch=[64] * 3, planes_trunk=[64] * 3).to(Device)
    elif 'FNO' in name:
        from fno.FNOs import FNO2d
        from run_FNO import inference
        Net_model = FNO2d(in_dim=in_dim, out_dim=out_dim, modes=mode, width=64, depth=4, steps=1,
                          padding=8, activation='gelu').to(Device)
    elif 'UNet' in name:
        from cnn.ConvNets import UNet2d
        from run_UNet import inference
        Net_model = UNet2d(in_sizes=(64, 64, 28), out_sizes=(64, 64, 5), width=64,
                           depth=4, steps=1, activation='gelu', dropout=0).to(Device)
    elif 'Transformer' in name:
        from basic.basic_layers import FcnSingle
        from fno.FNOs import FNO2d
        from transformer.Transformers import FourierTransformer
        from run_TransGV import inference, predictor

        with open(os.path.join("D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\transformer_config.yml")) as f:
            config = yaml.full_load(f)
            config = config['GV_RB']

            # 建立网络
        Tra_model = FourierTransformer(**config).to(Device)
        FNO_model = FNO2d(in_dim=2, out_dim=config['n_targets'], modes=(16, 16), width=64, depth=4,
                          padding=9, activation='gelu').to(Device)
        MLP_model = FcnSingle(planes=(in_dim, 64, 64, config['n_targets']), last_activation=True).to(Device)
        Net_model = predictor(trunc=FNO_model, branch=MLP_model, field_dim=out_dim).to(Device)

        # from transformer.Transformers import FourierTransformer2D
        # from run_Trans import inference
        # with open(os.path.join('transformer_config_sql.yml')) as f:
        #     config = yaml.full_load(f)
        #     config = config['Rotor37_2d']
        #     config['fourier_modes'] = mode
        # # 建立网络
        # Net_model = FourierTransformer2D(**config).to(Device)

    isExist = os.path.exists(os.path.join(work_path, 'latest_model.pth'))
    if isExist:
        checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'), map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])
        return Net_model, inference
    else:
        print("The pth file is not exist, CHECK PLEASE!")
        return None, None

def import_model_by_name(name):
    model_func = None
    inference = None
    train = None
    valid = None

    if 'MLP' in name:
        from run_MLP import MLP
        from run_MLP import inference, train, valid
        model_func = MLP
    elif 'deepONet' in name:
        from don.DeepONets import DeepONetMulti
        from Demo.GVRB_2d.run_model_GVRB.run_deepONet import inference, train, valid
        model_func = DeepONetMulti
    elif 'FNO' in name:
        from fno.FNOs import FNO2d
        from Demo.GVRB_2d.run_model_GVRB.run_FNO_struct import inference, train, valid
        model_func = FNO2d
    elif 'UNet' in name:
        from cnn.ConvNets import UNet2d
        from run_UNet import inference, train, valid
        model_func = UNet2d
    # elif 'Transformer' in name:
    #     from transformer.Transformers import FourierTransformer
    #     from run_TransGV import inference, train, valid
    #     model_func = FourierTransformer
    elif 'TNO' in name:
        from Tools.model_define.define_TNO import TransBasedNeuralOperator
        from Tools.model_define.define_TNO import inference, train, valid
        model_func = TransBasedNeuralOperator

    return model_func, inference, train, valid

def build_model_yml(yml_path, device, name=None):
    """
    build a new model based on yml file
    """
    # load the yml file
    with open(yml_path) as f:
        config = yaml.full_load(f)
        config = config[name + '_config'] #字典里面有字典
    # build the model
    model_func, inference, train, valid = import_model_by_name(name)
    # yml_path_gvrb = r"E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data\configs\transformer_config_gvrb.yml"
    net_model = model_func(**config).to(device)

    # net_model = FNO2d(in_dim=100, out_dim=8, modes=(4, 4), width=128, depth=4, steps=1,padding=8, activation='gelu').to(device)
    # net_model = model_func(**config,operator_dims=[100, ],planes_branch=[64] * 3, planes_trunk=[64] * 3 ).to(device)

    return net_model, inference, train, valid


def get_true_pred(loader, Net_model, inference, Device,
                  name=None, in_dim=100, out_dim=8, iters=0, alldata=False,
                  x_output=False,
                  ):
    true_list = []
    pred_list = []
    x_list = []
    set_size_sub  = 32
    if alldata:
        num = len(loader.dataset)
        iters = (num + set_size_sub -1)//set_size_sub

        new_loader = torch.utils.data.DataLoader(loader.dataset,
                                                 batch_size=set_size_sub,
                                                 shuffle=False,
                                                 drop_last=False)
        loader = new_loader

    for ii, data_box in enumerate(loader):
        if ii > iters:
            break
        if name in ("deepONet"):
            (data_x, data_f, data_y) = data_box
            sub_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_x, data_f, data_y),
                                                     batch_size=set_size_sub,
                                                     shuffle=False,
                                                     drop_last=False)
        else:
            (data_x, data_y) = data_box
            sub_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_x, data_y),
                                                     batch_size=set_size_sub,
                                                     shuffle=False,
                                                     drop_last=False)
    # for ii in range(iters):
        if name in ('MLP','TNO'):
            x, true, pred = inference(sub_loader, Net_model, Device)
        else:
            x, _, true, pred = inference(sub_loader, Net_model, Device)
        true = true.reshape([true.shape[0], 64, 128, out_dim])
        pred = pred.reshape([pred.shape[0], 64, 128, out_dim])
        if len(x.shape)>2:
            x = x[:,0,0,:]
        x = x.reshape([pred.shape[0], in_dim])
        # pred = pred.reshape([pred.shape[0], 32, 92])

        true_list.append(true)
        pred_list.append(pred)
        x_list.append(x)

    true = np.concatenate(true_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    x = np.concatenate(x_list, axis=0)

    if x_output:
        return x, true, pred
    else:
        return true, pred

if __name__ == "__main__":
    noi = get_noise([3,3], 0.1)
    print(noi)

    #输出预测结果