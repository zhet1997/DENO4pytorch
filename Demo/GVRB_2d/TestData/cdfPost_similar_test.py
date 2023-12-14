import numpy as np
import os
from Tools.post_process.post_CFD import cfdPost_2d
from scipy.interpolate import interp1d
# from utilizes_rotor37 import get_grid
import matplotlib.pyplot as plt
from Demo.GVRB_2d.utilizes_GVRB import get_grid, get_origin

def get_grid_interp(grid_num_s=128,
                    grid_num_z=128,
                    z_inlet=-0.059,
                    z_outlet=0.119,
                    hub_adjust=0.0001,
                    shroud_adjust=0.0001,
                    ):
    shroud_file = r"shroud_upper_GV.txt"
    hub_file = r"hub_lower_GV.txt"
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

if __name__ == "__main__":
    design, fields, grids = get_origin(
        type='struct',
        realpath=r'E:\\WQN\\CODE\\DENO4pytorch\\Demo\\GVRB_2d\\data',
        shuffled=False,
        getridbad=True,
        )

    # grid = get_grid_interp()
    # grid=get_grid(GV_RB=True, grid_num=128)
    post = cfdPost_2d(data=fields,
                      grid=grids,
                      boundarycondition=design[:,-4:],
                      similarity=False,
                      )
    # rst = post.get_performance('Static_pressure_ratio', z1=0, z2=127, type='spanwised')
    par_name = 'Absolute Total Pressure'
    bc_name = ''

    rst_1 = post.get_field_performance(par_name,type='averaged', z2=0).copy()
    bc_1 = post.bouCondition_1d[:, 2]
    # post.get_data_after_similarity(expand=2, scale=[-0.02,0.02])
    # rst_2 = post.get_field_performance(par_name, type='averaged', z2=0).copy()
    # bc_2 = post.bouCondition_1d[:, 2]

    # for ii in range(10):
        # plt.plot(rst_1[ii, :], np.linspace(0,1,64), c='r', linewidth=2)
        # plt.plot(rst_2[ii, :], np.linspace(0,1,64), c='b')

    ans = np.where(np.abs(rst_1[:, 0] - bc_1)>1e2)
    print(ans)
    plt.scatter(rst_1[:, 0], bc_1, c='r', s=20)
    # plt.scatter(rst_2[:, 0], bc_2, c='b', s=10)

    # plt.scatter(rst_1[:, 0], range(post.num_raw), c='r', s=20)
    # plt.scatter(rst_2[:, 0], range(post.num), c='b', s=10)

    plt.show()
    print(0)
