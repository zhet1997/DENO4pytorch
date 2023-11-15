import os
import torch
import yaml

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
        self.pth = os.path.join(self.root, 'latest_model.pth')
        self.pdparams = os.path.join(self.root, 'latest_model.pdparams')
        self.svg = os.path.join(self.root, 'log_loss.svg')
        self.yml= os.path.join(self.root, 'config.yml')
        self.x_norm = os.path.join(self.root, 'x_norm.pkl')
        self.y_norm = os.path.join(self.root, 'y_norm.pkl')

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