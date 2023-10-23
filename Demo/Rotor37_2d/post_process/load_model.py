import paddle
import os
import numpy as np
from Utilizes.utilizes_rotor37 import get_grid, get_origin
from Utilizes.process_data import DataNormer
import yaml

def get_noise(shape, scale):
    random_array = np.random.randn(np.prod(shape)) #  randn生成的是标准正态分布
    # random_array = (random_array-1)*2
    random_array = random_array.reshape(shape)

    return random_array * scale

def loaddata(name,
             ntrain=2500,
             nvalid=400,
             shuffled=False,
             noise_scale=None,
             loader=True,
             batch_size=32):

    design, fields = get_origin(realpath=os.path.join("..", "data"), shuffled=shuffled)  # 获取原始数据

    if name in ("FNO", "FNM", "UNet"):
        input = np.tile(design[:, None, None, :], (1, 64, 64, 1))
    else:
        input = design
    output = fields

    # input = paddle.to_tensor(input, dtype='float32')
    # output = paddle.to_tensor(output, dtype='float32')
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
    train_x = paddle.to_tensor(train_x, dtype='float32')
    train_y = paddle.to_tensor(train_y, dtype='float32')
    valid_x = paddle.to_tensor(valid_x, dtype='float32')
    valid_y = paddle.to_tensor(valid_y, dtype='float32')

    if name in ("deepONet"):
        grid = get_grid(realpath=os.path.join("..", "data"))
        grid_trans = paddle.to_tensor(grid[np.newaxis, :, :, :], dtype='float32')
        train_grid = paddle.tile(grid_trans, [train_x.shape[0], 1, 1, 1])  # 所有样本的坐标是一致的。
        valid_grid = paddle.tile(grid_trans, [valid_x.shape[0], 1, 1, 1])

        grid_normalizer = DataNormer(train_grid.numpy(), method='mean-std')  # 这里的axis不一样了
        train_grid = grid_normalizer.norm(train_grid)
        valid_grid = grid_normalizer.norm(valid_grid)

        # grid_trans = grid_trans.reshape([1, -1, 2])
        train_grid = train_grid.reshape([train_x.shape[0], -1, 2])
        valid_grid = valid_grid.reshape([valid_x.shape[0], -1, 2])
        train_y = train_y.reshape([train_y.shape[0], -1, 5])
        valid_y = valid_y.reshape([valid_y.shape[0], -1, 5])

        train_loader = paddle.io.DataLoader(paddle.io.TensorDataset([train_x, train_grid, train_y]),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = paddle.io.DataLoader(paddle.io.TensorDataset([valid_x, valid_grid, valid_y]),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = paddle.io.DataLoader(paddle.io.TensorDataset([train_x, train_y]),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = paddle.io.DataLoader(paddle.io.TensorDataset([valid_x, valid_y]),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

    if loader:
        return train_loader, valid_loader, x_normalizer, y_normalizer
    else:
        return train_x, train_y, valid_x, valid_y

def import_model_by_name(name):
    model_func = None
    inference = None
    train = None
    valid = None

    if 'MLP' in name:
        from model_define.define_MLP import MLP
        from model_define.define_MLP import inference, train, valid
        model_func = MLP
    elif 'deepONet' in name:
        from don.DeepONets import DeepONetMulti
        from model_define.define_deepONet import inference, train, valid
        model_func = DeepONetMulti
    elif 'FNO' in name:
        from fno.FNOs import FNO2d
        from model_define.define_FNO import inference, train, valid
        model_func = FNO2d
    elif 'UNet' in name:
        from cnn.ConvNets import UNet2d
        from model_define.define_UNet import inference, train, valid
        model_func = UNet2d
    elif 'TNO' in name:
        from model_define.define_TNO import TransBasedNeuralOperator
        from model_define.define_TNO import inference, train, valid
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
    net_model = model_func(**config).to(device)

    return net_model, inference, train, valid


def get_true_pred(loader, Net_model, inference, Device,
                  name=None, out_dim=5, iters=0, alldata=False):
    true_list = []
    pred_list = []
    set_size_sub  = 32
    if alldata:
        num = len(loader.dataset)
        iters = (num + set_size_sub -1)//set_size_sub

        new_loader = paddle.io.DataLoader(loader.dataset,
                                                 batch_size=set_size_sub,
                                                 shuffle=False,
                                                 drop_last=False)
        loader = new_loader

    for ii, data_box in enumerate(loader):
        if ii > iters:
            break
        if name in ("deepONet"):
            (data_x, data_f, data_y) = data_box
            sub_loader = paddle.io.DataLoader(paddle.io.TensorDataset([data_x, data_f, data_y]),
                                                     batch_size=set_size_sub,
                                                     shuffle=False,
                                                     drop_last=False)
        else:
            (data_x, data_y) = data_box
            sub_loader = paddle.io.DataLoader(paddle.io.TensorDataset([data_x, data_y]),
                                                     batch_size=set_size_sub,
                                                     shuffle=False,
                                                     drop_last=False)
    # for ii in range(iters):
        if name in ('MLP','Transformer'):
            _, true, pred = inference(sub_loader, Net_model, Device)
        else:
            _, _, true, pred = inference(sub_loader, Net_model, Device)
        true = true.reshape([true.shape[0], 64, 64, out_dim])
        pred = pred.reshape([pred.shape[0], 64, 64, out_dim])

        true_list.append(true)
        pred_list.append(pred)

    true = np.concatenate(true_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)

    return true, pred

if __name__ == "__main__":
    noi = get_noise([3,3], 0.1)
    print(noi)

    #输出预测结果