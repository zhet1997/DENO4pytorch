# running the cfd pred post process
import os
from matplotlib import pyplot as plt
from Utilizes.visual_data import MatplotlibVision
from post_process.model_predict import predictor_establish
from post_process.load_model import loaddata, get_true_pred
from post_process.post_data import Post_2d
from Utilizes.utilizes_rotor37 import get_grid
# load the model
if __name__ == "__main__":
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", 'work', 'model_save')
    grid = get_grid(real_path=os.path.join("..", "data"))
    Net_model, inference, Device, x_normalizer, y_normalizer = \
        predictor_establish(name, work_load_path, predictor=False)

    predictor = predictor_establish(name, work_load_path)

    ## predict the valid samples
    train_loader, valid_loader, _, _ = loaddata(name, 2500, 400, shuffled=True)
    train_x, train_y, valid_x, valid_y = loaddata(name, 2500, 400, shuffled=True, loader=False)

    true, pred = get_true_pred(valid_loader, Net_model, inference, Device,
                               name=name, iters=0, alldata=False)
    pred_2 = predictor.predicter_2d(valid_x[:5,0,0,:], input_norm=True) # the input has already normalized


    true = y_normalizer.back(true)
    pred = y_normalizer.back(pred)

    post_true = Post_2d(true, grid)
    post_pred = Post_2d(pred, grid)

    parameterList = [
        "PressureTotalV",
        "TemperatureTotalV",
        "EntropyStatic",
        "MachIsentropic",
        "Load"
    ]

    for kk, parameter in enumerate(parameterList):
        true[..., kk] = getattr(post_true, parameter)
        pred[..., kk] = getattr(post_pred, parameter)

    # dict = plot_error(post_true, post_pred, parameterList + ["MassFlow"],
    #            paraNameList=parameterListN,
    #            save_path=None, fig_id=0, label=None, work_path=work_path, type=type)

    # if type=="valid":
    #     np.savez(os.path.join(work_path, "FNO_num.npz"), **dict)

    # plot_field_2d(post_true, post_pred, parameterList, work_path=work_path, type=type, grid=grid)
    save_path = ''
    Visual = MatplotlibVision(save_path, input_name=('x', 'y'), field_name=('p', 't', 'V', 'W', 'mass'))
    # field_name = ['p[kPa]', 't[K]', '$\mathrm{v^{2}/1000[m^{2}s^{-2}]}$', '$\mathrm{w^{2}/1000[m^{2}s^{-2}]}$', '$\mathrm{mass [kg m^{-2}s^{-1}]}$']
    field_name = ['p[kPa]', 't[K]',
                  '$\mathrm{S[J/K]}$',
                  '$\mathrm{Ma_{is}}$',
                  '$\mathrm{\psi}$']

    # scaleList = [1000, 1, 1, 1, 1]
    scaleList = [1000, 1, 1000, 1000, 1]
    limitList = [20, 20, 50, 0.1, 0.1]

    for ii in [5, 8, 58]:
        for jj in range(5):
            true[ii, :, :, jj:jj + 1] = true[ii, :, :, jj:jj + 1] / scaleList[jj]
            pred[ii, :, :, jj:jj + 1] = pred[ii, :, :, jj:jj + 1] / scaleList[jj]

            # fig, axs = plt.subplots(3, 1, figsize=(30, 6), num=jj)
            # Visual.field_name = [field_name[jj]]
            # Visual.plot_fields_ms_col(fig, axs, (true[ii,:,:,jj:jj+1]), (pred[ii,:,:,jj:jj+1])
            #                           , grid, show_channel=[0],cmaps = ['Spectral_r', 'Spectral_r', 'coolwarm'],
            #                           limit=limitList[jj])
        # fig, axs = plt.subplots(5, 3, figsize=(25, 15), num=1)

        fig, axs = plt.subplots(5, 3, figsize=(45, 30), num=1)
        Visual.field_name = field_name
        Visual.plot_fields_ms(fig, axs, true[ii], pred[ii],
                              grid, show_channel=None, cmaps=['Spectral_r', 'Spectral_r', 'coolwarm'],
                              limitList=limitList)

        save_path = os.path.join("..", "data", "final_fig")
        fig.patch.set_alpha(0.)
        plt.show()

        # fig.savefig(os.path.join(save_path, 'derive' + str(0) + '_field_' + str(ii) + '.png'), transparent=True)
        # plt.close(fig)





