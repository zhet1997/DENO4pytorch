import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import matplotlib.pyplot as plt
from TestData.post_CFD import cfdPost_2d
from scipy.interpolate import interp1d
from utilizes_rotor37 import get_grid, get_origin, get_origin_GVRB
from Tools.post_process.load_model import loaddata_Sql, get_true_pred
from Tools.post_process.model_predict import predictor_establish

def plot_spanwise_distribution(rst_true, rst_pred, num, save_name, save_path):
    font0 = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'normal', 'size': 18}
    font1 = {'family': 'Times New Roman', 'size': 15}

    plt.figure(figsize=(4.5, 4), dpi=300)
    ax0 = plt.subplot(111)
    plt.plot(rst_true[num,:], np.linspace(0,1,128), ls='-', lw=2,
             color='blue', label='CFD')
    plt.plot(rst_pred[num,:], np.linspace(0,1,128), ls='--', lw=2,
             color='red', label='Prediction')
    plt.xlabel("Isentropic efficiency", fontsize=18, fontdict=font0)
    plt.ylabel("Spanwise", fontsize=18, fontdict=font0)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # plt.grid(linestyle = '--', linewidth = 2.0)
    labels_0 = ax0.get_xticklabels() + ax0.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels_0]
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.legend(prop=font1, frameon=False, fontsize='large', loc='best')
    ax00 = plt.gca()
    bwith = 1.0
    ax00.spines['bottom'].set_linewidth(bwith)  # 图框下边
    ax00.spines['left'].set_linewidth(bwith)  # 图框左边
    ax00.spines['top'].set_linewidth(bwith)  # 图框上边
    ax00.spines['right'].set_linewidth(bwith)  # 图框右边
    plt.tight_layout()
    plt.savefig(save_path +'/'+ save_name +'.jpg', dpi=200)
    plt.clf()



if __name__ == '__main__':
    design, fields = get_origin_GVRB(quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                   "Vx", "Vy", "Vz",
                                                   'Relative Total Temperature',
                                                   'Absolute Total Temperature'])

    # name = 'Transformer'
    # input_dim = 96
    # output_dim = 8
    # ## load the model
    # print(os.getcwd())
    # work_load_path = os.path.join("..", 'work_Trans5000_2')
    # grid = get_grid(real_path='D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\TestData', GV_RB=True)
    # Net_model, inference, Device, x_normalizer, y_normalizer = \
    #     predictor_establish(name, work_load_path, predictor=False)
    #
    #
    # pathNpz = os.path.join('D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\save_valid.npz')
    # data = np.load(pathNpz)
    # # data_true = data['true']
    # data_pred = data['pred']
    #
    # train_loader, valid_loader, _, _, save_x, data_true = \
    #     loaddata_Sql('Transformer',
    #                  4000, 1000, shuffled=True,
    #                  norm_x=x_normalizer,
    #                  norm_y=y_normalizer,
    #                  )
    # output = fields
    data_true = fields[0:20,...]
    data_pred = fields[19:89,...]



    inputdict = {'Static Pressure': 0,
                 'Static Temperature': 1,
                 'Density': 2,
                 'Vx': 3,
                 'Vy': 4,
                 'Vz': 5,
                 'Relative Total Temperature': 6,
                 'Absolute Total Temperature': 7,
                 }

    # variable_name = 'Isentropic_efficiency'
    variable_name = 'Static_pressure_ratio'
    grid = get_grid(GV_RB=True, grid_num=128)
    post_true = cfdPost_2d(data_true, grid, inputDict=inputdict)
    rst_true = post_true.get_performance('{}'.format(variable_name), type='spanwised')
    print(rst_true.shape)
    plt.plot(rst_true[0, :], np.linspace(0, 1, 128))
    plt.show()
    post_pred = cfdPost_2d(data_pred, grid, inputDict=inputdict)
    rst_pred = post_pred.get_performance('{}'.format(variable_name), type='spanwised')
    print(rst_pred.shape)
    # save_path = os.path.join("D:/WQN/CODE/DENO4pytorch-main/Demo/GV_RB/work_Trans5000_2/Transformer/result/")
    # plot_spanwise_distribution(rst_true, rst_pred, 0, '0', save_path)

