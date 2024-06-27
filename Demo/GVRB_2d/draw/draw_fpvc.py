import os
from Tools.draw_figure.draw_uncetainty import draw_uq_2d
from Tools.draw_figure.draw_uncetainty import draw_uq_0d, draw_uq_0d_multi_bc, draw_uq_1d
from Tools.optimization.pymoo_optimizer import predictor_establish
from Tools.post_process.model_predict import DLModelPost
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
from Tools.draw_figure.draw_opt_rst import optiaml_predicter
import matplotlib.pyplot as plt
from draw_opt import get_compare_data, load_opt_data

if __name__ == '__main__':
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'TNO'
    input_dim = 100
    output_dim = 8
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    var_name = ['S1', 'R1']
    parameterList = [
        # 'Static_pressure_ratio',
        'Total_total_efficiency',
        # 'Total_static_efficiency',
        'Degree_reaction',
        'Mass_flow'
        # 'atan(Vx/Vz)',
        # 'atan(Wx/Wz)',
    ]

    uqList = ['tangle', 'ttem', 'tpre', 'rotate']
    cmap = plt.cm.get_cmap('tab10')
    colorList = cmap.colors

    # X = np.array([0.5] * 48)

    X = get_compare_data(var_name)

    # X, _, _ = load_opt_data(var_name, type='base')

    model_all = predictor_establish(name, work_load_path)
    draw_uq_all()