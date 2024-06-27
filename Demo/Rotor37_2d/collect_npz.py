# 将不同模型对于valid的预测值整理到一起来
from draw_figure.utilizes_draw import *
import os

def get_dict(name, work):
    # nameReal = name.split("_")[0]
    nameReal = "Transformer"
    id = None
    # if len(name.split("_")) == 2:
    #     id = int(name.split("_")[1])

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
    if os.path.exists(work.yml):
        Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=nameReal)
        isExist = os.path.exists(work.pth)
        if isExist:
            checkpoint = torch.load(work.pth, map_location=Device)
            Net_model.load_state_dict(checkpoint['net_model'])
    else:
        Net_model, inference = rebuild_model(work_path, Device, name=nameReal)
    train_loader, valid_loader, _, _ = loaddata(nameReal, 2500, 400, shuffled=True)

    for type in ["valid"]:
        if type == "valid":
            true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                                       name=nameReal, iters=10, alldata=True)
        elif type == "train":
            true, pred = get_true_pred(train_loader, Net_model, inference, Device,
                                       name=nameReal, iters=10, alldata=True)
        true = y_normalizer.back(true)
        pred = y_normalizer.back(pred)

        input_para = {
            "PressureStatic": 0,
            "TemperatureStatic": 1,
            "V2": 2,
            "W2": 3,
            "DensityFlow": 4,
        }

        grid = get_grid(real_path="D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\data")
        plot_error_box(true, pred, save_path=None, type=type, work_path=work_path)

        post_true = Post_2d(true, grid,
                            inputDict=input_para,
                            )
        post_pred = Post_2d(pred, grid,
                            inputDict=input_para,
                            )

        # parameterList = []
        parameterList = [
            "Efficiency", "EfficiencyPoly",
            "PressureRatioV", "TemperatureRatioV",
            "PressureLossR", "EntropyStatic",
            "MachIsentropic",
            "Load",
        ]
        parameterListN = [
            "PR", "TR",
            "Eff", "EffPoly",
            "PLoss", "Entr",
            "Mach", "Load",
            "MF"]


    dict = plot_error(post_true, post_pred, parameterList + ["MassFlow"],
                      paraNameList=parameterListN,
                      save_path=None, fig_id=0, label=None, work_path=work_path, type=type)

    return dict


if __name__ == "__main__":
    input_dim = 28
    output_dim = 5
    save_path = os.path.join("..", "data", "surrogate_data")
    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load",
        "MassFlow"]

    load_path_list = [
                    "work",
                    "work",
                    "work",
                    "work",
                    "work",
                     ]

    file_name_list_num = [
                    "Trunk_TRA500",
                    "Trunk_TRA1000",
                    "Trunk_TRA1500",
                    "Trunk_TRA2000",
                    "Trunk_TRA",
                     ]

    file_name_list_noise = [
                    "Trunk_TRA500",
                    "Trunk_TRA1000",
                    "Trunk_TRA1500",
                    "Trunk_TRA2000",
                    "Trunk_TRA",
                     ]
    dict_num = {}
    dict_noise = {}
    # num_list = [4, 0, 1, 2, 3]
    # noise_list = [0, 1, 2, 3, 4]
    num_list = [0, 1, 2, 3, 4]
    for parameter in parameterList:
        # dict_noise.update({parameter : np.zeros([400, 5])})
        dict_num.update({parameter: np.zeros([400, 5])})

    for ii, (name, file) in enumerate(zip(file_name_list_num, load_path_list)):
        work_load_path = os.path.join("..", file)
        work_path = os.path.join(work_load_path, name)
        work = WorkPrj(work_path)

        dict = get_dict(name, work)


        for parameter in parameterList:
            dict_num[parameter][:,num_list[ii]:num_list[ii]+1] = dict[parameter]


    np.savez(os.path.join(save_path, "Trunk_num.npz"), **dict_num)

    # for ii, (name, file) in enumerate(zip(file_name_list_noise, load_path_list)):
    #     work_load_path = os.path.join("..", file)
    #     work_path = os.path.join(work_load_path, name)
    #     work = WorkPrj(work_path)
    #
    #     dict = get_dict(name, work)
    #
    #     for parameter in parameterList:
    #         dict_noise[parameter][:,noise_list[ii]:noise_list[ii]+1] = dict[parameter]
    #
    # np.savez(os.path.join(save_path, "MLP_noise.npz"), **dict_noise)