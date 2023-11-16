import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import matplotlib.pyplot as plt
from TestData.post_CFD import cfdPost_2d
# from scipy.interpolate import interp1d
from Demo.GVRB_2d.utilizes_GVRB import get_grid, get_origin
# from Tools.post_process.load_model import loaddata_Sql, get_true_pred
# from Tools.post_process.model_predict import predictor_establish

def plot_spanwise_distribution(rst_true, rst_pred, num, variable_name, save_name, save_path):
    font0 = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'normal', 'size': 18}
    font1 = {'family': 'Times New Roman', 'size': 15}

    plt.figure(figsize=(4.5, 4), dpi=300)
    ax0 = plt.subplot(111)
    plt.plot(rst_true[num,:], np.linspace(0,1,64), ls='-', lw=2,
             color='blue', label='CFD')
    plt.plot(rst_pred[0,:], np.linspace(0,1,128), ls='--', lw=2,
             color='red', label='Prediction')
    plt.xlabel(variable_name, fontsize=18, fontdict=font0)
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
    plt.close()



if __name__ == '__main__':
    design, fields, grids = get_origin(type='struct',realpath='..\data', shuffled=False,
                                       quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                   "Vx", "Vy", "Vz",
                                                   'Relative Total Temperature',
                                                   'Absolute Total Temperature'])
    inputdict = {'Static Pressure': 0,
                 'Static Temperature': 1,
                 'Density': 2,
                 'Vx': 3,
                 'Vy': 4,
                 'Vz': 5,
                 'Relative Total Temperature': 6,
                 'Absolute Total Temperature': 7,
                 }
    save_path = "E://WQN//CODE//DENO4pytorch//Demo//GVRB_2d//work//TNO_2//result_1"
    # [0, 1, 2, 3, 6, 7, 8, 9][13,14,15,16,17]
    # for num in np.array([13,14,15,16,17]):
    #     pathNpz = os.path.join('test_case5\Case{}\sampleRstZip.npz'.format(num))
    #     data = np.load(pathNpz)
    #     data_2d = np.zeros([1,128,128,8])
    #     for key in inputdict.keys():
    #         for ii in range(1):
    #             data_2d[ii:ii+1,:,:,inputdict[key]] = data[key]
    #
    #     for key in inputdict.keys():
    #         save_name = '{}_{}_outlet'.format(key, num)
    #         plt.figure(figsize=(4.5, 4), dpi=300)
    #         plt.plot(fields[num, :, 127, inputdict[key]], np.linspace(0,1,64),ls='-',c='blue')
    #         # plt.plot(fields[0, :, 0, 0], np.linspace(0, 1, 64))
    #         plt.plot(data_2d[0,:,127,inputdict[key]], np.linspace(0,1,128),ls='--',c='orange')
    #         plt.tight_layout()
    #         plt.savefig(save_path + '//' + save_name + '.jpg', dpi=200)
    #         plt.close()

    data_true = fields[:20,:,:,:]
    # variable_name_list = ['Total_total_efficiency','Total_static_efficiency','Static_pressure_ratio', 'Enthalpy', 'Degree_reaction']
    variable_name_list = [ 'Total_static_efficiency', 'Static_pressure_ratio', 'Degree_reaction', 'Total_total_efficiency', ]
    # variable_name = 'Static_pressure_ratio'
    # grid = get_grid(GV_RB=True, grid_num=128)
    post_true = cfdPost_2d(data_true, grids, inputDict=inputdict)
    for variable_name in variable_name_list:
        rst_true = post_true.get_performance('{}'.format(variable_name), type='spanwised')
        print(rst_true.shape)
        for num in np.array([0, 1, 2, 3, 6, 7, 8, 9]):
            pathNpz = os.path.join('test_case10\Case_{}\sampleRstZip.npz'.format(num))
            data = np.load(pathNpz)
            data_2d = np.zeros([1,128,128,8])
            for key in inputdict.keys():
                for ii in range(1):
                    data_2d[ii:ii+1,:,:,inputdict[key]] = data[key]
            post_2d = cfdPost_2d(data_2d, grids, inputDict=inputdict)
            rst_2d = post_2d.get_performance('{}'.format(variable_name), type='spanwised')
            print(rst_2d.shape)
            save_name = '{}_{}'.format(variable_name, num)
            plot_spanwise_distribution(rst_true, rst_2d, num, variable_name, save_name, save_path)

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


    # train_loader, valid_loader, _, _, save_x, data_true = \
    #     loaddata_Sql('Transformer',
    #                  4000, 1000, shuffled=True,
    #                  norm_x=x_normalizer,
    #                  norm_y=y_normalizer,
    #                  )
    # output = fields