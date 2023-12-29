import sys
sys.path.insert(0, r'E:\WQN\CODE\DENO4pytorch')  # 替换为你的项目根目录的实际路径
#==========#
import numpy as np
from Tools.post_process.post_CFD import cfdPost_2d
from scipy.interpolate import interp1d
from Demo.GVRB_2d.utilizes_GVRB import get_origin
from Tools.post_process.load_model import loaddata_Sql
from Utilizes.process_data import DataNormer

def get_grid_interp(grid_num_s=128,
                    grid_num_z=128,
                    z_inlet=-0.059,
                    z_outlet=0.119,
                    hub_adjust=0.0001,
                    shroud_adjust=0.0001,
                    ):
    shroud_file = r"../../../Demo/GVRB_2d/data/check/shroud_upper_GV.txt"
    hub_file = r"../../../Demo/GVRB_2d/data/check/hub_lower_GV.txt"
    hub = np.loadtxt(hub_file)/1000
    shroud = np.loadtxt(shroud_file)/1000

    x = np.linspace(z_inlet, z_outlet, grid_num_z)
    xx = np.tile(x, [grid_num_s, 1])

    f_hub = interp1d(hub[:, 0], hub[:, 1], kind='linear')
    y_hub = f_hub(x)
    f_shroud = interp1d(shroud[:, 0], shroud[:, 1], kind='linear')
    y_shroud = f_shroud(x)

    yy = []
    for i in range(grid_num_z):
        yy.append(np.linspace(y_hub[i]+hub_adjust,y_shroud[i]-shroud_adjust,grid_num_s)) # check

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(grid_num_z, grid_num_s).T
    xx = xx.reshape(grid_num_s, grid_num_z)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)

def test_similarity(par_name='Isentropic_efficiency',
                    bc_name=None,
                    ):
    design, fields, grids = get_origin(
        type='struct',
        realpath=r'E:\\WQN\\CODE\\DENO4pytorch\\Demo\\GVRB_2d\\data',
        shuffled=False,
        getridbad=True,
        )

    post = cfdPost_2d(data=fields,
                      grid=grids,
                      boundarycondition=design[:,-4:],
                      similarity=False,
                      )

    rst_1 = post.get_field_performance(par_name,type='averaged').copy()
    bc_1 = post.bouCondition_1d[:, 2]
    perf_1 = post.get_field_performance(par_name).copy()

    post.get_data_after_similarity(expand=1, scale=[-0.02,0.02]) # make the similarity change
    rst_2 = post.get_field_performance(par_name, type='averaged').copy()
    bc_2 = post.bouCondition_1d[:, 2]
    perf_2 = post.get_field_performance(par_name).copy()


    assert np.all((perf_1-perf_2)<1e-5)


    # for ii in range(10):
    #     plt.plot(rst_1[ii, :], np.linspace(0,1,64), c='r', linewidth=2)
    #     plt.plot(rst_2[ii, :], np.linspace(0,1,64), c='b')
    #
    # plt.scatter(rst_2[:, 0], bc_2, c='b', s=10)
    # plt.scatter(rst_1[:, 0], range(post.num_raw), c='r', s=20)
    # plt.scatter(rst_2[:, 0], range(post.num), c='b', s=10)
    #
    # plt.show()
    # print(0)
def test_loader_similarity(par_name_1='Isentropic_efficiency',
                           par_name_2='Mass_flow',
                            bc_name=None,
                            ):
    _, _, grids = get_origin(
        type='struct',
        realpath=r'E:\\WQN\\CODE\\DENO4pytorch\\Demo\\GVRB_2d\\data',
        shuffled=False,
        getridbad=True,
    )

    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata_Sql('TNO', 4000, 700, shuffled=True,)
    x_normalizer_bc = x_normalizer
    x_normalizer_bc.shrink(slice(96, 100, 1))



    post = cfdPost_2d()
    post.loader_readin(valid_loader,
                        grid=grids,
                       x_norm=x_normalizer_bc,
                       y_norm=y_normalizer,
                       )
    perf_raw_1 = post.get_field_performance(par_name_1).copy()
    perf_raw_2 = post.get_field_performance(par_name_2).copy()

    post = cfdPost_2d()
    valid_loader_sim = post.loader_similarity(valid_loader,
                                                grid=grids, scale=[-0.01, 0.01], expand=1, log=True,
                                                x_norm=x_normalizer_bc,
                                                y_norm=y_normalizer,
                                               )

    post_sim = cfdPost_2d()
    post_sim.loader_readin(valid_loader_sim,
                       grid=grids,
                       x_norm=x_normalizer_bc,
                       y_norm=y_normalizer,
                           )

    perf_new_1 = post_sim.get_field_performance(par_name_1).copy()
    perf_new_2 = post_sim.get_field_performance(par_name_2).copy()

    assert np.all((perf_new_1 - perf_raw_1) < 1e-5)
    assert np.any(np.abs(perf_new_2 - perf_raw_2) > 1e-3)
    print(0)


def test_loader_log_similarity(par_name_1='Isentropic_efficiency',
                           par_name_2='Mass_flow',
                            bc_name=None,
                            ):
    designs, fields, grids = get_origin(
        type='struct',
        realpath=r'E:\\WQN\\CODE\\DENO4pytorch\\Demo\\GVRB_2d\\data',
        shuffled=False,
        getridbad=True,
    )

    x_normalizer = DataNormer(designs, method='log')
    x_normalizer.dim_change(2)


    y_normalizer = DataNormer(fields, method='log')
    y_normalizer.dim_change(2)

    train_loader, valid_loader, x_normalizer, y_normalizer = loaddata_Sql('TNO', 4000, 700, shuffled=True,
                                                                          norm_x=x_normalizer,
                                                                          norm_y=y_normalizer,
                                                                          )
    x_normalizer_bc = x_normalizer
    x_normalizer_bc.shrink(slice(96, 100, 1))



    post = cfdPost_2d()
    post.loader_readin(valid_loader,
                        grid=grids,
                       x_norm=x_normalizer_bc,
                       y_norm=y_normalizer,
                       )
    perf_raw_1 = post.get_field_performance(par_name_1).copy()
    perf_raw_2 = post.get_field_performance(par_name_2).copy()

    post = cfdPost_2d()
    valid_loader_sim = post.loader_similarity(valid_loader,
                                                grid=grids, scale=[-0.01, 0.01], expand=1, log=True,
                                                x_norm=x_normalizer_bc,
                                                y_norm=y_normalizer,
                                               )

    post_sim = cfdPost_2d()
    post_sim.loader_readin(valid_loader_sim,
                       grid=grids,
                       x_norm=x_normalizer_bc,
                       y_norm=y_normalizer,
                           )

    perf_new_1 = post_sim.get_field_performance(par_name_1).copy()
    perf_new_2 = post_sim.get_field_performance(par_name_2).copy()

    assert np.all((perf_new_1 - perf_raw_1) < 1e-5)
    assert np.any(np.abs(perf_new_2 - perf_raw_2) > 1e-3)
    print(0)

if __name__ == "__main__":
    test_similarity()


