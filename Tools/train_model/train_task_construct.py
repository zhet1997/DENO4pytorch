import copy
import os
import torch
import yaml
import numpy as np
from Tools.post_process.post_CFD import cfdPost_2d
def work_construct(para_list_dict):
    work_list = []
    for key in para_list_dict.keys():
        num = len(para_list_dict[key])
        for ii in range(num):
            work_list.append({key:para_list_dict[key][ii]})

    return work_list

def work_construct_togethor(para_list_dict):
    work_list = []
    num = 1
    for key in para_list_dict.keys():
        num = num * len(para_list_dict[key])

    for ii in range(num):
        dict_new = {}
        for key in para_list_dict.keys():
            idx = ii % len(para_list_dict[key])
            dict_new.update({key:para_list_dict[key][idx]})
        work_list.append(dict_new)
    return work_list

def change_yml(name, yml_path=None, **kwargs):
    # 加载config模板
    template_path = os.path.join("..", "data", "config_template.yml")
    with open(template_path, 'r', encoding="utf-8") as f:
        config_all = yaml.full_load(f)
        config_para = config_all[name + '_config']
    # 修改参数
    for key in kwargs.keys():
        if key in config_para.keys():
            config_para[key] = kwargs[key]
        else:
            print("The keywords {} is illegal, CHECK PLEASE".format(key))
    # 保存到新的文件
    isExist = os.path.exists(yml_path)
    if not isExist:
        with open(yml_path, 'w', encoding="utf-8") as f:
            pass
    with open(yml_path, 'r', encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if data is None:
        data = {}
    data[name + '_config'] = config_para
    with open(yml_path, 'w', encoding="utf-8") as f:
        yaml.dump(data, f)

def add_yml(key_set_list, yml_path=None):
    template_path = os.path.join("..", "data", "config_template.yml")
    with open(template_path, 'r', encoding="utf-8") as f:
    # 加载config模板
        config_all = yaml.full_load(f)
    for key_set in key_set_list:
        config_para = config_all[key_set]
    # 保存到新的文件
        with open(yml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        data[key_set] = config_para
        with open(yml_path, 'w') as f:
            yaml.dump(data, f)

def feature_transform(x):
    """
    Args:
        x: input coordinates
    Returns:
        res: input transform
    """
    shape = x.shape
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.linspace(0, 1, size_x, dtype=torch.float32)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.linspace(0, 1, size_y, dtype=torch.float32)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

    edge = torch.ones((x.shape[0], 1))
    return torch.cat((gridx, gridy), dim=-1).to(x.device), edge.to(x.device)



class WorkPrj(object):
    def __init__(self, work_path):
        self.root = work_path
        isExist = os.path.exists(self.root)

        if not isExist:
            os.mkdir(self.root)
        name = os.path.basename(self.root)
        name = name.split("_")[0]
        self.name = name
        self.pth = os.path.join(self.root, 'latest_model.pth')
        self.fpth = os.path.join(self.root, 'final_model.pth')
        self.pdparams = os.path.join(self.root, 'latest_model.pdparams')
        self.svg = os.path.join(self.root, 'log_loss.svg')
        self.yml= os.path.join(self.root, 'config.yml')
        self.x_norm = os.path.join(self.root, 'x_norm.pkl')
        self.y_norm = os.path.join(self.root, 'y_norm.pkl')
        self.train = os.path.join(self.root, 'train.npz')
        self.valid = os.path.join(self.root, 'valid.npz')
        self.valid_sim = os.path.join(self.root, 'valid_sim.npz')
        self.grid = os.path.join(self.root, 'grid.npz')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device

    def config(self, name):
        with open(self.yml) as f:
            config_all = yaml.full_load(f)

        if name + "_config" in config_all.keys():
            return config_all[name + "_config"]
        else:
            return None

    def get_dict(self, npz_path=None):
        if os.path.exists(npz_path):
            dict = np.load(npz_path)
            return dict
        else:
            print('the file {} is not exist!'.format(npz_path))
            return False

    def save_pred(self):
        from Tools.post_process.model_predict import predictor_establish
        from Tools.post_process.load_model import loaddata_Sql, get_true_pred
        Net_model, inference, Device, x_normalizer, y_normalizer = \
            predictor_establish(self.name, self.root, is_predictor=False)
<<<<<<< HEAD
        train_loader, valid_loader, _, _ = loaddata_Sql(self.name, 4000, 900, shuffled=True,
=======
        train_loader, valid_loader, _, _ = loaddata_Sql(self.name, 500, 200, shuffled=True,
>>>>>>> b408901fbe28cc64936b95ecdbce7d553f1e42ca
                                                        norm_x=x_normalizer, norm_y=y_normalizer)


        for data in ['train', 'valid']:
            if data=='train':
                loader = train_loader
                path_save = self.train
            else:
                loader = valid_loader
                path_save = self.valid

            x, true, pred = get_true_pred(loader, Net_model, inference, Device,
                                       self.name, iters=0, alldata=True, out_dim=8, in_dim=100, x_output=True)
            x = x_normalizer.back(x)
            true = y_normalizer.back(true)
            pred = y_normalizer.back(pred)

            save_dict = {}
            save_dict.update({'x': x})
            save_dict.update({'true': true})
            save_dict.update({'pred': pred})
            np.savez(path_save, **save_dict)

        post = cfdPost_2d()
        norm_bc = copy.deepcopy(x_normalizer)
        norm_bc.shrink(slice(96, 100, 1))
        valid_loader_sim = post.loader_similarity(valid_loader,
                                                  scale=[-0.005, 0.005], expand=1, log=True,
                                                  x_norm=norm_bc,
                                                  y_norm=y_normalizer,
                                                  )
        x, true, pred = get_true_pred(valid_loader_sim, Net_model, inference, Device,
                                      self.name, iters=0, alldata=True, out_dim=8, in_dim=100, x_output=True)
        x = x_normalizer.back(x)
        true = y_normalizer.back(true)
        pred = y_normalizer.back(pred)

        save_dict = {}
        save_dict.update({'x': x})
        save_dict.update({'true': true})
        save_dict.update({'pred': pred})
        np.savez(self.valid_sim, **save_dict)



