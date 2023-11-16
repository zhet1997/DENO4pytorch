# running the cfd pred post process
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
from matplotlib import pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.post_process.model_predict import predictor_establish
from Tools.post_process.load_model import loaddata_Sql, get_true_pred
from Tools.post_process.post_data import Post_2d
from utilizes_rotor37 import get_grid, get_origin_GVRB

if __name__ == "__main__":
    name = 'TNO_2'
    input_dim = 100
    output_dim = 8
    ## load the model
    print(os.getcwd())
    work_load_path = os.path.join("..", 'work')
    grid = get_grid(real_path='E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\data')
    Net_model, inference, Device, x_normalizer, y_normalizer = \
        predictor_establish(name, work_load_path, predictor=False)
    ## predict the valid samples
    name = 'TNO'
    train_loader, valid_loader, _, _ = loaddata_Sql(name, 4000, 1000, shuffled=True,
                                                    norm_x=x_normalizer, norm_y=y_normalizer)
    true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                               name=name, iters=0, alldata=True)
    true = y_normalizer.back(true)
    pred = y_normalizer.back(pred)

    save_dict={}
    save_dict.update({'true': true})
    save_dict.update({'pred': pred})
    np.savez(os.path.join('..','save_valid.npz'),**save_dict)






