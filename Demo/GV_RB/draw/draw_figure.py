from utilizes_draw import *
import yaml
from transformer.Transformers import FourierTransformer
from run_TransGV import inference
from model_whole_life import WorkPrj
from TestData.post_CFD import cfdPost_2d
from utilizes_rotor37 import get_grid, get_origin_GVRB


if __name__ == "__main__":
    parameterList = ['Isentropic efficiency',
                     ]

    inputdict = {
                 "Static Pressure":0,
                 "Static Temperature":1,
                 "Density":2,
                  "Vx":3, "Vy":4, "Vz":5,
                  'Relative Total Temperature':6,
                  'Absolute Total Temperature':7
                 }


    name = 'Transformer'
    input_dim = 96
    output_dim = 8
    ntrain = 4000
    nvalid = 1000
    batch_size = 32

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

        design, fields = get_origin_GVRB(quanlityList=["Static Pressure", "Static Temperature", "Density",
                                                       "Vx", "Vy", "Vz",
                                                       'Relative Total Temperature',
                                                       'Absolute Total Temperature'])
        input = np.tile(design[:, None, None, :], (1, 128, 128, 1))
        input = torch.tensor(input, dtype=torch.float)

        # output = fields[:, 0, :, :, :].transpose((0, 2, 3, 1))
        output = fields
        output = torch.tensor(output, dtype=torch.float)

        print(input.shape, output.shape)

        train_x = input[:ntrain, :, :]
        train_y = output[:ntrain, :, :]
        valid_x = input[ntrain:ntrain + nvalid, :, :]
        valid_y = output[ntrain:ntrain + nvalid, :, :]

        x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
        train_x = x_normalizer.norm(train_x)
        valid_x = x_normalizer.norm(valid_x)

        y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
        train_y = y_normalizer.norm(train_y)
        valid_y = y_normalizer.norm(valid_y)

        x_normalizer.save(os.path.join(work_path, 'x_norm.pkl'))  # 将normalizer保存下来
        y_normalizer.save(os.path.join(work_path, 'y_norm.pkl'))

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                                   batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),
                                                   batch_size=batch_size, shuffle=False, drop_last=True)

        with open(r"D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\transformer_config_8.yml") as f:
            config = yaml.full_load(f)
            config = config['GV_RB']
            # config['modes'] = 4

        Net_model = FourierTransformer(**config).to(Device)
        Net_model = Net_model.to(Device)

        work_pth = 'D:\WQN\CODE\DENO4pytorch-main\Demo\GV_RB\work_Trans5000_2\Transformer\latest_model.pth'
        isExist = os.path.exists(work_pth)
        if isExist:
            checkpoint = torch.load(work_pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])

        Net_model.eval()
        # train_loader, valid_loader, _, _ = loaddata_Sql(name, **work.config("Basic"))

        for type in ["valid", "train"]:
            if type == "valid":
                true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                           name=nameReal, iters=5, alldata=False)
            elif type == "train":
                true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                           name=nameReal, iters=5, alldata=False)
        # true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
        #                            name=nameReal, iters=5, alldata=False
        #                            )
            true = y_normalizer.back(true)
            pred = y_normalizer.back(pred)

            grid = get_grid(GV_RB=True, grid_num=128)

            post_true = cfdPost_2d(true, grid, inputDict=inputdict)
            post_pred = cfdPost_2d(pred, grid, inputDict=inputdict)


            for parameter_Name in parameterList:
                save_path = os.path.join("..", "..", "work_Trans5000_2", "Transformer")
                plot_error(post_true, post_pred, parameter_Name,
                               save_path= save_path,fig_id=0,
                               label=None, work_path = work_path, type=type, paraNameList=None)

