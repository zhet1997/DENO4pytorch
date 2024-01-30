import numpy as np
from numpy import ndarray
from Utilizes.process_data import DataNormer
import torch
def data_padding(data: ndarray, const=0, channel_num: int=16, expand=True, shuffle=True) -> ndarray:
    # the padding is only work for the last channel dim
    current_channel_num = data.shape[-1]
    if expand:
        while channel_num < current_channel_num:
            channel_num = channel_num * 2

    assert channel_num >= current_channel_num
    input = np.zeros([*data.shape[:-1], channel_num], dtype=np.float32) + const
    input[..., :current_channel_num] = data

    if shuffle:
        for ii in range(input.shape[0]):
            idx = np.random.permutation(input.shape[-1])
            input[ii] = input[ii, :, :, idx].transpose(1,2,0)

    return input

def split_train_valid(data,
                    train_num=0, valid_num=0,
                    train_ratio=0, valid_ratio=0,
                    method='num'
                     ):
    data_num = data.shape[0]
    if method=='num':
        # assert train_num + valid_num < data_num, print('Too many train and valid samples, CHECK PLEASE !!!')

        if train_num + valid_num > data_num: #test only
            train = data[0:1]
        else:
            train = data[:train_num]
        valid = data[-valid_num:]
        return train, valid

def channel_to_instance(data, channel_num=1, list=True):
    assert data.shape[-1]%channel_num==0, print('Wrong channel_num, CHECK PLEASE !!!')
    # assert torch.is_tensor(data), print('It is for tensor only, CHECK PLEASE !!!')
    data_new = [data[..., i:i + channel_num] for i in range(0, int(data.shape[-1]), channel_num)]
    if not list:
        if torch.is_tensor(data):
            data_new = torch.cat(data_new, dim=0)
        else:
            data_new = np.concatenate(data_new, axis=0)
    return data_new

def instance_to_half(data, batch_size=1, list=True):
    assert data.shape[0]%batch_size==0, print('Wrong channel_num, CHECK PLEASE !!!')
    assert torch.is_tensor(data), print('It is for tensor only, CHECK PLEASE !!!')
    data_new = [data[i:i + int(data.shape[0]/2), ...] for i in range(0, int(data.shape[0]), int(data.shape[0]/2))]
    if not list:
        data_new = torch.cat(data_new, dim=-1)
    return data_new


def fill_channels(xx, x_norm=None, channel_num=16, shuffle=False,):
    expand = int(channel_num - xx.shape[-1])
    xx_fill = torch.zeros([*xx.shape[:-1], expand], device=xx.device) + 350
    xx_fill = x_norm.norm(xx_fill)
    xx = torch.cat((xx, xx_fill), dim=-1)  # now the channel num is 16*(2**super_num)

    if shuffle:
        for ii in range(xx.shape[0]):
            idx = torch.randperm(xx.shape[-1])
            xx[ii] = xx[ii, ..., idx]

    return xx

def little_windows(ff, num_rows=4, num_cols=4):
    (b_z, f_rows, f_cols, c) = ff.shape
    ff = (ff.reshape(b_z, f_rows//num_rows, num_rows, f_cols//num_rows, num_cols, c)
          .permute(0,2,4,1,3,5)
          .reshape(b_z*num_rows*num_cols, f_rows//num_rows, f_cols//num_rows, c))
    return ff

def big_windows(ff, num_rows=4, num_cols=4):
    (b_z, f_rows, f_cols, c) = ff.shape
    ff = (ff.reshape(b_z//num_rows//num_cols, num_rows, num_cols, f_rows, f_cols, c)
          .permute(0,3,1,4,2,5)
          .reshape(b_z//num_rows//num_cols, f_rows*num_rows, f_cols*num_rows, c))
    return ff

def get_loader_from_list(
            train_x_list,
            train_y_list,
            x_normalizer=None,
            y_normalizer=None,
            combine_list=True,
            batch_size = 32,
            shuffle=True,
           ):

    if combine_list:
        train_x = np.concatenate(train_x_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)

    if x_normalizer is None:
        assert combine_list
        x_normalizer = DataNormer(train_x, method='mean-std', axis=(0,1,2,3))
    if y_normalizer is None:
        assert combine_list
        y_normalizer = DataNormer(train_y, method='mean-std')

    if combine_list:
        train_x = x_normalizer.norm(train_x)
        train_y = y_normalizer.norm(train_y)

        train_x = torch.as_tensor(train_x, dtype=torch.float)
        train_y = torch.as_tensor(train_y, dtype=torch.float)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        train_loader = []
        for train_x, train_y in zip(train_x_list, train_y_list):
            train_x = x_normalizer.norm(train_x)
            train_y = y_normalizer.norm(train_y)

            train_x = torch.as_tensor(train_x, dtype=torch.float)
            train_y = torch.as_tensor(train_y, dtype=torch.float)

            train_loader.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                       batch_size=batch_size, shuffle=shuffle, drop_last=True))


    return train_loader, x_normalizer, y_normalizer


def get_loader_from_unpadding_list(
            data_x_list,
            data_y_list,
            x_normalizer=None,
            y_normalizer=None,
            combine_list=True,
            batch_size = 32,
            shuffle=True,
            channel_num=10,
            padding=True,
           ):

    data_x_all = []
    if padding:
        for data_x in data_x_list:
            data_x_all.append(data_padding(data_x, const=350, channel_num=channel_num, shuffle=False))
        data_x_list = data_x_all
        del data_x_all

    if x_normalizer is None:
        x_normalizer = norm_for_list(data_x_list)
    if y_normalizer is None:
        y_normalizer = norm_for_list(data_y_list)

    if combine_list:
        data_x = x_normalizer.norm(np.concatenate(data_x_list, axis=0))
        data_y = y_normalizer.norm(np.concatenate(data_y_list, axis=0))

        data_x = torch.as_tensor(data_x, dtype=torch.float)
        data_y = torch.as_tensor(data_y, dtype=torch.float)
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_x, data_y),
                                                   batch_size=batch_size, shuffle=shuffle, drop_last=True)
    else:
        train_loader = []
        for train_x, train_y in zip(data_x_list, data_y_list):
            train_x = x_normalizer.norm(train_x)
            train_y = y_normalizer.norm(train_y)

            train_x = torch.as_tensor(train_x, dtype=torch.float)
            train_y = torch.as_tensor(train_y, dtype=torch.float)

            train_loader.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                       batch_size=batch_size, shuffle=shuffle, drop_last=True))


    return train_loader, x_normalizer, y_normalizer

def get_loader_from_list_combine(
            data_x_list,
            data_y_list,
            x_normalizer=None,
            y_normalizer=None,
            batch_size = 32,
            shuffle=True,
            channel_num=16,
           ):

    if x_normalizer is None:
        x_normalizer = norm_for_list(data_x_list)
    if y_normalizer is None:
        y_normalizer = norm_for_list(data_y_list)

    def collate_fn(batch):
        x_list = []
        y_list = []
        for ii in range(len(batch)):
            current_channel_size = batch[ii][0].size(-1)
            if current_channel_size < channel_num:
                x_list.append(torch.unsqueeze(fill_channels(batch[ii][0], x_norm=x_normalizer, channel_num=channel_num, shuffle=True), dim=0))
                y_list.append(torch.unsqueeze(batch[ii][1], dim=0))

        x_list = torch.cat(x_list, dim=0)
        y_list = torch.cat(y_list, dim=0)

        return (x_list, y_list)

    dataset_list = []
    for data_x, data_y in zip(data_x_list, data_y_list):
        data_x = x_normalizer.norm(data_x)
        data_y = y_normalizer.norm(data_y)

        data_x = torch.as_tensor(data_x, dtype=torch.float)
        data_y = torch.as_tensor(data_y, dtype=torch.float)
        dataset_list.append(torch.utils.data.TensorDataset(data_x, data_y))
    combined_dataset = torch.utils.data.ConcatDataset(dataset_list)

    train_loader = torch.utils.data.DataLoader(combined_dataset,
                                               collate_fn = collate_fn,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               drop_last=True)

    # aaa = next(iter(train_loader))

    return train_loader, x_normalizer, y_normalizer

def norm_for_list(data_list, method='mean-std', axis=None):
    combine_list = []
    for data in data_list:
        combine_list.append(channel_to_instance(data, channel_num=1, list=False))
    combine_data = np.concatenate(combine_list, axis=0)
    normalizer = DataNormer(combine_data, method='mean-std')

    return normalizer


if __name__ == "__main__":
    pass
