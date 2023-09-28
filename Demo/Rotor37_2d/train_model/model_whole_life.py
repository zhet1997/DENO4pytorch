import os
import paddle
import numpy as np
from Utilizes.utilizes_rotor37 import Rotor37WeightLoss
from train_task_construct import WorkPrj, add_yml, change_yml
from post_process.load_model import build_model_yml, loaddata
from post_process.model_predict import DLModelPost
from Utilizes.visual_data import MatplotlibVision
import matplotlib.pyplot as plt
import yaml
import time

class DLModelWhole(object):
    def __init__(self, device,
                 name=None,
                 in_norm=None,
                 out_norm=None,
                 grid_size=64,
                 work=None,
                 epochs=1000,
                 ):
        self.device = device
        self.work = work
        self.name = name
        self.net_model, self.inference, self.train, self.valid = \
            build_model_yml(work.yml, self.device, name=name)

        self.in_norm = in_norm
        self.out_norm = out_norm
        self.grid_size = grid_size

        self.epochs = epochs

        self.Loss_func = None
        self.Optimizer = None
        self.Scheduler = None

    def set_los(self):
        with open(self.work.yml) as f:
            config = yaml.full_load(f)
        # 损失函数
        # self.Loss_func = paddle.nn.MSELoss()
        self.Loss_func = Rotor37WeightLoss()
        # 优化算法
        self.Scheduler = paddle.optimizer.lr.StepDecay(**config["Scheduler_config"])
        self.Optimizer = paddle.optimizer.Momentum(learning_rate=self.Scheduler, parameters=self.net_model.parameters())

    def train_epochs(self, train_loader, valid_loader):
        work = self.work
        Visual = MatplotlibVision(work.root, input_name=('x', 'y'), field_name=('unset',))
        star_time = time.time()
        log_loss = [[], []]
        for epoch in range(self.epochs):
            self.net_model.train()
            log_loss[0].append(self.train(train_loader, self.net_model, self.device, self.Loss_func, self.Optimizer, self.Scheduler))
            self.net_model.eval()
            log_loss[1].append(self.valid(valid_loader,self.net_model, self.device, self.Loss_func))
            print('epoch: {:6d}, lr: {:.3e}, train_step_loss: {:.3e}, valid_step_loss: {:.3e}, cost: {:.2f}'.
                  format(epoch, self.Optimizer.get_lr(), log_loss[0][-1], log_loss[1][-1], time.time() - star_time))
            # print(os.environ['CUDA_VISIBLE_DEVICES'])
            star_time = time.time()

            if epoch > 0 and epoch % 5 == 0:
                fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[0, :], 'train_step')
                Visual.plot_loss(fig, axs, np.arange(len(log_loss[0])), np.array(log_loss)[1, :], 'valid_step')
                fig.suptitle('training loss')
                fig.savefig(work.svg)
                plt.close(fig)

            if epoch > 0 and epoch % 100 == 0:
                paddle.save(
                    {'log_loss': log_loss, 'net_model': self.net_model.state_dict(), 'optimizer': self.Optimizer.state_dict()},
                    work.pth)


if __name__ == "__main__":
    name = "MLP"
    id = 0
    train_num = 2500
    valid_num = 450
    work = WorkPrj(os.path.join("..", "work_train", name + "_" + str(id)))

    if paddle.device.is_compiled_with_cuda():
        Device = paddle.device.set_device('gpu')
    else:
        Device = paddle.device.set_device('cpu')

    config_dict = {
                    'n_hidden': 512,
                    'num_layers': 10,
                  }
    change_yml(name, yml_path=work.yml, **config_dict)
    add_yml(["Optimizer_config", "Scheduler_config"], yml_path=work.yml)
    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, train_num, valid_num, shuffled=True)
    x_normalizer.save(work.x_norm)
    y_normalizer.save(work.y_norm)
    DL_model = DLModelWhole(Device, name=name, work=work)
    DL_model.set_los()
    DL_model.train_epochs(train_loader, valid_loader)

    post = DLModelPost(DL_model.net_model, Device, name=name, in_norm=x_normalizer, out_norm=y_normalizer)

    Rst = []
    for batch, (input, output) in enumerate(valid_loader):
        Rst.append(post.predictor_value(input, parameterList="PressureLossR", input_norm=True))

    Rst = np.concatenate(Rst, axis=1)