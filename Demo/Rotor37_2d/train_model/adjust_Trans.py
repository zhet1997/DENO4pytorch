import os
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from post_process.post_data import Post_2d
from run_FNO import feature_transform
from Demo.Rotor37_2d.utilizes_rotor37 import get_grid
from post_process.load_model import build_model_yml, loaddata
from post_process.model_predict import DLModelPost
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
import yaml
import time
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml

def work_construct(para_list_dict):
    work_list = []
    for key in para_list_dict.keys():
        num = len(para_list_dict[key])
        for ii in range(num):
            work_list.append({key:para_list_dict[key][ii]})

    return work_list



if __name__ == "__main__":
    name = "Transformer"
    batch_size = 32
    start_id = 0
    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')
    train_num = 2500
    valid_num = 400
    dict_model = {
                'num_encoder_layers': [3, 5, 6],
                'n_hidden': [64, 128, 256],
                'dropout': [0.5],
                }
    dict_optimizer = {
        'learning_rate': [0.1, 0.08, 0.06, 0.05]
    }

    model_list = work_construct(dict_model)
    opt_list = work_construct(dict_optimizer)

    for id, config_dict in enumerate(opt_list):
        work = WorkPrj(os.path.join("..", "work_train_1", name + "_" + str(id + start_id)))

        # change_yml(name, yml_path=work.yml, **config_dict)
        # add_yml(["Optimizer_config", "Scheduler_config"], yml_path=work.yml)

        change_yml("Optimizer", yml_path=work.yml, **config_dict)
        add_yml([name + "_config", "Scheduler_config"], yml_path=work.yml)

        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, train_num, valid_num, shuffled=True)
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(Device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)
