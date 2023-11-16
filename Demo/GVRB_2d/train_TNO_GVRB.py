# running the cfd pred post process
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(torch.cuda.is_available())
import numpy as np
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from Tools.post_process.model_predict import predictor_establish
from Tools.post_process.load_model import loaddata_Sql, get_true_pred
from Tools.train_model.train_task_construct import WorkPrj
from utilizes_rotor37 import get_grid, get_origin_GVRB
from Tools.model_define.define_TNO import train, valid, inference

if __name__ == "__main__":
    name = 'TNO_3'
    input_dim = 96
    output_dim = 8
    ## load the model
    print(os.getcwd())
    work_load_path = os.path.join('work')
    work = WorkPrj(os.path.join(work_load_path, name))
    grid = get_grid(real_path='D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\TestData',GV_RB=True)
    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata_Sql(name, 4000, 700, shuffled=True, )
    x_normalizer.save(work.x_norm)
    y_normalizer.save(work.y_norm)
    Net_model, inference, Device, _, _ = \
        predictor_establish(name, work_load_path, predictor=False)
    ## predict the valid samples

    batch_size = 32
    epochs = 1001
    learning_rate = 0.001
    scheduler_step = 800
    scheduler_gamma = 0.1


    Loss_func = nn.MSELoss()
    # 优化算法
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    # 下降策略
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # 可视化
    Visual = MatplotlibVision('', input_name=('x', 'y'),
                              field_name=('ps', 'ts', 'rho', 'vx', 'vy', 'vz', 'tt1', 'tt2'))
    star_time = time.time()
    log_loss = [[], []]

    ################################################################
    # train process
    ################################################################
    # grid = get_grid()
    grid = get_grid(GV_RB=True, grid_num=128)
    for epoch in range(epochs):

        Net_model.train()
        log_loss[0].append(train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler))

        Net_model.eval()
        log_loss[1].append(valid(valid_loader, Net_model, Device, Loss_func))
        print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
              format(epoch, learning_rate, log_loss[0][-1], log_loss[1][-1], time.time() - star_time))

        star_time = time.time()

        if epoch > 0 and epoch % 10 == 0:
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
            Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
            fig.suptitle('training loss')
            fig.savefig(os.path.join(work.root, 'log_loss.svg'))
            plt.close(fig)

        ################################################################
        # Visualization
        ################################################################

        if epoch % 100 == 0:
            train_source, train_true, train_pred = inference(train_loader, Net_model, Device)
            valid_source, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
            torch.save({'log_loss': log_loss, 'net_model': Net_model.state_dict(), 'optimizer': Optimizer.state_dict()},
                       os.path.join(work.root, 'latest_model.pth'))
        #
            for fig_id in range(5):
                fig, axs = plt.subplots(8, 3, figsize=(18, 25), num=2)
                Visual.plot_fields_ms(fig, axs, train_true[fig_id], train_pred[fig_id], grid)
                fig.savefig(os.path.join(work.root, 'train_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)

            for fig_id in range(5):
                fig, axs = plt.subplots(8, 3, figsize=(18, 25), num=3)
                Visual.plot_fields_ms(fig, axs, valid_true[fig_id], valid_pred[fig_id], grid)
                fig.savefig(os.path.join(work.root, 'valid_solution_' + str(fig_id) + '.jpg'))
                plt.close(fig)
        #





