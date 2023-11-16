import numpy as np
import os

from Utilizes.visual_data import MatplotlibVision

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import matplotlib.pyplot as plt
from TestData.post_CFD import cfdPost_2d
# from scipy.interpolate import interp1d
from Demo.GVRB_2d.utilizes_GVRB import get_grid, get_origin
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import mean_squared_error as MSE, r2_score
from Tools.post_process.model_predict import predictor_establish
# from Tools.post_process.load_model import loaddata_Sql, get_true_pred
# from Tools.post_process.model_predict import predictor_establish

def plot_spanwise_distribution(rst_true, rst_pred, num, variable_name, save_name, save_path):
    font0 = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'normal', 'size': 14}
    font1 = {'family': 'Times New Roman', 'size': 10}

    plt.figure(figsize=(3, 4), dpi=300)
    ax0 = plt.subplot(111)
    plt.plot(rst_true[num,:], np.linspace(0,1,64), ls='-', lw=1.5, alpha=0.8,
             color='blue', label='CFD')
    plt.scatter(rst_pred[num,:], np.linspace(0,1,64), s=3,
             color='red', label='Prediction')
    plt.xlabel(variable_name.replace('_',' '), fontdict=font0)
    plt.ylabel("Spanwise", fontdict=font0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.grid(linestyle = '--', linewidth = 2.0)
    labels_0 = ax0.get_xticklabels() + ax0.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels_0]
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.legend(prop=font1, frameon=False, loc='best')
    ax00 = plt.gca()
    bwith = 1.0
    ax00.spines['bottom'].set_linewidth(bwith)  # 图框下边
    ax00.spines['left'].set_linewidth(bwith)  # 图框左边
    ax00.spines['top'].set_linewidth(bwith)  # 图框上边
    ax00.spines['right'].set_linewidth(bwith)  # 图框右边
    plt.tight_layout()
    plt.savefig(save_path +'/'+ save_name +'.jpg', dpi=200)
    plt.close()

def plot_performance_compare(rst_true, rst_pred, variable_name, save_name, save_path):
    middle = (np.max(rst_true) - np.min(rst_true))/2
    plt.figure(figsize=(4.5, 4))
    ax = plt.subplot(111)
    plt.scatter(rst_true, rst_pred, color='red', s=2)
    plt.xlabel('CFD', fontsize=14, fontproperties='Times New Roman')
    plt.ylabel('Predictionc', fontsize=14, fontproperties='Times New Roman')
    plt.text(np.min(rst_true)+0.005, np.max(rst_true)-0.005, 'R\u00b2=%.3f' % (r2_score(rst_true, rst_pred)), fontsize=14
             , fontproperties='Times New Roman')
    plt.text(np.max(rst_true)-middle, np.max(rst_true)-middle+0.01, '+0.006', fontsize=12, rotation=45, fontproperties='Times New Roman')
    plt.text(np.min(rst_true)+middle, np.min(rst_true)+middle-0.01, '-0.006', fontsize=12, rotation=45, fontproperties='Times New Roman')
    plt.tick_params(labelsize=15)
    labels_1 = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels_1]

    plt.xlim(np.min(rst_true)-0.01, np.max(rst_true)+0.005)
    plt.ylim(np.min(rst_true)-0.01, np.max(rst_true)+0.005)
    xlim = np.arange(np.min(rst_true)-0.01, np.max(rst_true)+0.005, 0.001)
    plt.plot(plt.xlim(np.min(rst_pred)-0.01, np.max(rst_pred)+0.005),
             plt.ylim(np.min(rst_pred)-0.01, np.max(rst_pred)+0.005), ls="--", c="black")
    plt.fill_between(xlim, xlim + 0.006, xlim - 0.006, facecolor='C0', alpha=0.2)
    plt.grid(linestyle='--', linewidth=1.0)
    bwith = 1.0
    ax.spines['bottom'].set_linewidth(bwith)  # 图框下边
    ax.spines['left'].set_linewidth(bwith)  # 图框左边
    ax.spines['top'].set_linewidth(bwith)  # 图框上边
    ax.spines['right'].set_linewidth(bwith)  # 图框右边
    plt.tight_layout()
    plt.savefig(save_path +'/'+ save_name +'.jpg')
    plt.close()



if __name__ == '__main__':
    design, fields, grids = get_origin(type='struct',realpath='..\data',
                                       quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                   "Vx", "Vy", "Vz",
                                                   'Relative Total Temperature',
                                                   'Absolute Total Temperature'])

    pathNpz = os.path.join('E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work\TNO_3\save_valid.npz')
    data = np.load(pathNpz)
    data_true = data['true']
    data_pred = data['pred']

    inputdict = {'Static Pressure': 0,
                 'Static Temperature': 1,
                 'Density': 2,
                 'Vx': 3,
                 'Vy': 4,
                 'Vz': 5,
                 'Relative Total Temperature': 6,
                 'Absolute Total Temperature': 7,
                 }
    # variable_name_list = ['Total_total_efficiency', 'Total_static_efficiency', 'Static_pressure_ratio', 'Degree_reaction']
    post_true = cfdPost_2d(data_true, grids, inputDict=inputdict)
    post_pred = cfdPost_2d(data_pred, grids, inputDict=inputdict)

    #比较沿叶高分布
    # variable_name_list = ['Total_total_efficiency', 'Total_static_efficiency', 'Static_pressure_ratio']
    # for variable_name in variable_name_list:
    #     rst_true = post_true.get_performance('{}'.format(variable_name), type='spanwised')
    #     print(rst_true.shape)
    #
    #     rst_pred = post_pred.get_performance('{}'.format(variable_name), type='spanwised')
    #     print(rst_pred.shape)
    #
    #     save_path = "E://WQN//CODE//DENO4pytorch//Demo//GVRB_2d//work//TNO_2//result_spanwise"
    #     for num in range(10):
    #         save_name = '{}_{}'.format(variable_name, num)
    #         plot_spanwise_distribution(rst_true, rst_pred, num, variable_name, save_name, save_path)

    #比较性能预测结果
    variable_name_list = ['Total_total_efficiency', 'Total_static_efficiency']
    for variable_name in variable_name_list:
        rst_true = post_true.get_performance('{}'.format(variable_name), type='averaged')
        print(rst_true.shape)

        rst_pred = post_pred.get_performance('{}'.format(variable_name), type='averaged')
        print(rst_pred.shape)

        save_path = "E://WQN//CODE//DENO4pytorch//Demo//GVRB_2d//work//TNO_3//result_averaged"
        save_name = '{}'.format(variable_name)
        plot_performance_compare(rst_true, rst_pred, variable_name, save_name, save_path)

    #比较子午流面流场云图预测结果
    # Visual = MatplotlibVision(work_load_path, input_name=('x', 'y'), field_name=(
    #     'Absolute Total Pressure', 'Absolute Mach Number','Static Enthalpy'))
    # for fig_id in range(5):
    #     fig, axs = plt.subplots(8, 3, figsize=(30, 40), num=2)
    #     Visual.plot_fields_ms(fig, axs, true[fig_id], pred[fig_id], grid)
    #     fig.savefig(os.path.join(work_load_path, 'solution_' + type + "_" + str(fig_id) + '.jpg'))
    #     plt.close(fig)

