from utilizes_draw import *
from Tools.post_process.load_model import loaddata_Sql
import yaml
from transformer.Transformers import FourierTransformer
from run_TransGV import inference
from model_whole_life import WorkPrj

if __name__ == "__main__":

    name = 'Transformer'
    input_dim = 96
    output_dim = 4
    work_path = os.path.join("D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_Trans5000_2")
    workList = os.listdir(work_path)
    folder_template = 'Transformer_{}'
    for i in [0]:
        # 根据i值创建文件夹名称
        work_name = folder_template.format(i)
        work_name = "Transformer"
        # 构建文件夹的完整路径
        work_load_path = os.path.join(work_path, work_name)
        work = WorkPrj(work_load_path)
        name_parts = work_name.split("_")
        nameReal = name_parts[0]

        if torch.cuda.is_available():
            Device = torch.device('cuda')
        else:
            Device = torch.device('cpu')

        norm_save_x = work.x_norm
        norm_save_y = work.y_norm

        x_normalizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
        x_normalizer.load(norm_save_x)
        y_normalizer = DataNormer(np.ndarray([1, 1]), method="mean-std", axis=0)
        y_normalizer.load(norm_save_y)


        # with open('D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_FNO1\FNO_2\config.yml') as f:

        # with open('D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_MLP1\MLP_0\config.yml') as f:
        with open(r"D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\transformer_config_8.yml") as f:
            config = yaml.full_load(f)
            config = config['GV_RB']
            # config['modes'] = 4



        # 建立网络
        ##only MLP
        # layer_mat = [92, 256, 256, 256, 256, 512, 4 * 128 * 128]
        # Net_model = MLP(layers=layer_mat, is_BatchNorm=False)
        # Net_model = Net_model.to(Device)

        # Net_model = FourierTransformer(**config).to(Device)
        # Net_model = FNO2d(in_dim=92, out_dim=4, modes=(4,4), width=128, depth=4, steps=1,
        #                   padding=8, activation='relu').to(Device)

        Net_model = FourierTransformer(**config).to(Device)
        Net_model = Net_model.to(Device)

        work_pth = 'D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_Trans5000_2\Transformer\latest_model.pth'
        isExist = os.path.exists(work_pth)
        if isExist:
            checkpoint = torch.load(work_pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])

        Net_model.eval()
        train_loader, valid_loader, _, _ = loaddata_Sql(name, **work.config("Basic"))

        for type in ["valid","train"]:
            if type == "valid":
                true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                           name=nameReal, iters=1, alldata=False)
            elif type == "train":
                true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                           name=nameReal, iters=1, alldata=False)

            true = y_normalizer.back(true)
            pred = y_normalizer.back(pred)

            grid = get_grid(GV_RB=True, grid_num=128)

            parameterList = [
                "Static Pressure", "Static Temperature", "Density",
                                                       "Vx", "Vy", "Vz",
                                                       'Relative Total Temperature',
                                                       'Absolute Total Temperature']




            # Visual = MatplotlibVision(work_load_path, input_name=('x', 'y'), field_name=('Static Pressure', 'Static Temperature','Absolute Total Temperature', 'DensityFlow'))
            #
            # for fig_id in range(5):
            #     fig, axs = plt.subplots(4, 3, figsize=(30, 40), num=2)
            #     Visual.plot_fields_ms(fig, axs, true[fig_id], pred[fig_id], grid)
            #     fig.savefig(os.path.join(work_load_path, 'solution_' + type + "_" + str(fig_id) + '.jpg'))
            #     plt.close(fig)


            ## only MLP
            # train_coord, train_true, train_pred = inference(train_loader, Net_model, Device)
            # valid_coord, valid_true, valid_pred = inference(valid_loader, Net_model, Device)
            Visual = MatplotlibVision(work_load_path, input_name=('x', 'y'), field_name=('Static Pressure', 'Static Temperature','Absolute Total Temperature', 'DensityFlow'))
            for fig_id in range(5):
                fig, axs = plt.subplots(8, 3, figsize=(30, 40), num=2)
                Visual.plot_fields_ms(fig, axs, true[fig_id], pred[fig_id], grid)
                fig.savefig(os.path.join(work_load_path, 'solution_' + type + "_" + str(fig_id) + '.jpg'))
                plt.close(fig)
