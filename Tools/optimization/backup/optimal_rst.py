# 将优化预测结果与CFD验证结果对比
from Tools.optimization.backup.CFD_verify import load_CFD_mat
from Tools.draw_figure.backup.draw_sensitive import predicter_loader
# from draw_figure.utilizes_draw import plot_span_curve_marker
# from train_model.model_whole_life import WorkPrj
# from Utilizes.process_data import DataNormer
# from post_process.load_model import build_model_yml
# from Demo.Rotor37_2d.utilizes_rotor37 import get_grid, get_origin
# from post_process.post_data import Post_2d
from Tools.draw_figure.backup.utilizes_draw import *

def get_predict(work_path, input, name, Device, parameterList):
    parameter_dict = {}
    work = WorkPrj(work_path)

    norm_save_x = work.x_norm
    norm_save_y = work.y_norm

    x_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    x_normlizer.load(norm_save_x)
    y_normlizer = DataNormer([1, 1], method="mean-std", axis=0)
    y_normlizer.load(norm_save_y)

    Net_model, inference, _, _ = build_model_yml(work.yml, Device, name=name)
    isExist = os.path.exists(work.pth)
    if isExist:
        checkpoint = torch.load(work.pth, map_location=Device)
        Net_model.load_state_dict(checkpoint['net_model'])

    Net_model.eval()

    input = x_normlizer.norm(input)
    pred = predicter_loader(Net_model, input, Device, name=name)

    pred = pred.reshape([pred.shape[0], 64, 64, 5])
    pred = y_normlizer.back(pred)

    input_para = {
        "PressureStatic": 0,
        "TemperatureStatic": 1,
        "V2": 2,
        "W2": 3,
        "DensityFlow": 4,
    }
    grid = get_grid(real_path=os.path.join("../../../Demo/Rotor37_2d", "data"))
    post_pred = Post_2d(pred.detach().cpu().numpy(), grid,
                        inputDict=input_para,
                        )

    for parameter in parameterList:
        if parameter=="MassFlow":
            scalar = post_pred.get_MassFlow()
        else:
            value_span = getattr(post_pred, parameter)
            scalar = post_pred.span_density_average(value_span[:, :, -1])
        parameter_dict.update({parameter: scalar})

    return parameter_dict



def find_closest_row(A, B):
    min_distance = np.inf
    closest_row_idx = None

    for i, row in enumerate(A):
        distance = np.linalg.norm(row - B)
        if distance < min_distance:
            min_distance = distance
            closest_row_idx = i

    return closest_row_idx, min_distance


if __name__ == "__main__":
    work_path = os.path.join("../../../Demo/Rotor37_2d", "work_train_FNO2", "FNO_1") #当前效果最好的模型
    name_opt = "EPM_optimization_tasks"
    name = "FNO"

    if torch.cuda.is_available():
        Device = torch.device('cuda')
    else:
        Device = torch.device('cpu')

    data_path = os.path.join("../../../Demo/Rotor37_2d", "data")
    mat_path = os.path.join(data_path, "sampleRstZip_optimal.mat")




    #加载CFD结果
    parameterList = [
        "PressureRatioV", "TemperatureRatioV",
        "PressureRatioW", "TemperatureRatioW",
        "Efficiency", "EfficiencyPoly",
        "PressureLossR", "EntropyStatic",
        "MachIsentropic", "Load", "MassFlow"]

    dict_true = load_CFD_mat(mat_path, parameterList)
    dict_base = load_CFD_mat(os.path.join("../../../Demo/Rotor37_2d", "data", "sampleRstZip_57"), parameterList)

    dataBox = np.zeros([len(parameterList), 4])
    #加载预测结果 # 需要加载模型获得全部结果
    true = {}
    true_base = {}
    for ii, parameter in enumerate(parameterList):
        true.update({parameter: dict_true[parameter]})
        true_base.update({parameter: dict_base[parameter][0]})
        dataBox[ii] = dict_true[parameter].T

    save_path = os.path.join("../../../Demo/Rotor37_2d", "data", "final_fig", 'optimal_rst.csv')
    np.savetxt(save_path, dataBox, delimiter=',')

    "PressureRatioV",
    "TemperatureRatioV",
    "PressureRatioW",
    "TemperatureRatioW",
    "Efficiency",
    "EfficiencyPoly",
    "PressureLossR",
    "EntropyStatic",
    "MachIsentropic",
    "Load",
    "MassFlow"






