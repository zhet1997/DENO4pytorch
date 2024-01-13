import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.train_model.train_task_construct import WorkPrj
from Tools.post_process.post_CFD import cfdPost_2d
from Demo.GVRB_2d.utilizes_GVRB import get_grid_interp,get_origin
from Utilizes.process_data import DataNormer

def plot_loss(work_path, save_path):

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    fig, axs = plt.subplots(1, 1, figsize=(12, 6), num=1)
    colors = plt.cm.get_cmap('Dark2').colors[:5]
    checkpoint = torch.load(os.path.join(work_path, 'latest_model.pth'), map_location=Device)
    log_loss = checkpoint['log_loss']
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('unset'))
    # tmp = log_loss[4].copy()
    # log_loss[4] = log_loss[3].copy()
    # log_loss[3] = log_loss[2].copy()
    # log_loss[2] = tmp
    # log_loss = log_loss[(0,1,3,2,4)]

    lab_name = ('train','valid_origin','valid_expand','similitude','regularization')
    for tt in range(len(log_loss)):
        num = 1
        loss_box = np.zeros([num,len(log_loss[1])])
        loss_box[0] = log_loss[tt]
        for jj in range(1,num):
            loss_box[jj, jj:] = log_loss[1][:-jj]
            #buzu
            loss_box[jj,:jj] = np.tile(log_loss[1][0],[1,jj])

        normalizer = DataNormer(loss_box, method='mean-std', axis=0)
        normalizer.std = np.clip(normalizer.std, 0, normalizer.std[1])
        Visual.plot_value_std_clean(fig, axs, np.arange(len(log_loss[1])), normalizer.mean,
                                    std=normalizer.std, stdaxis=1, rangeIndex=0.95,
                                    log=True, label=lab_name[tt],
                                    title=None, xylabels=("epoch", "loss value"),color=colors[tt],)



        # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :] * scaleList[ii],
        #                  'train_' + nameList[ii], color=colors[ii], linestyle='--')
        # Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :] * scaleList[ii],
        #                  'valid_' + nameList[ii], color=colors[ii], linestyle='-')
        # fig.suptitle('training loss')

    # axs.legend(loc="best", ncol=2)
    # Visual.font["size"] = 11
    axs.set_xlim(0, 500)
    axs.set_ylim(7e-5, 0.5)
    axs.legend(loc="best", framealpha=1, prop=Visual.font)

    fig.savefig(os.path.join(save_path, 'log_loss_std.jpg'))
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':


    # name = 'TNO_7'
    # name = 'FNO__batch_num_5.3'
    # name = 'FNO_4_self_combine_reg_5.3.3_expand_2'
    # name = 'FNO_8_self_combine_reg'
    name = 'FNO_0'
    input_dim = 100
    output_dim = 8
    type = 'valid'
    stage_name = 'stage'
    print(os.getcwd())
    work_load_path = os.path.join("E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d", 'work')
    work = WorkPrj(os.path.join(work_load_path, name))
    save_path = os.path.join(work.root, 'save_figure')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plot_loss(work.root, save_path)