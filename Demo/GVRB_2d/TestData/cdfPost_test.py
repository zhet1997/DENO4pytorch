import numpy as np
import os
from Tools.post_process.post_CFD import cfdPost_2d
from scipy.interpolate import interp1d
from utilizes_rotor37 import get_grid
import matplotlib.pyplot as plt

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
    pathNpz = os.path.join('sampleRstZip.npz')
    data = np.load(pathNpz)
    inputdict = {'Static Pressure':0,
                 'Absolute Total Pressure':1,
                 'Static Temperature':2,
                 'Absolute Total Temperature':3,
                 'Vx':4,
                 'Vy':5,
                 'Vz':6,
                 'Density':7,
                 }
    data_2d = np.zeros([10,128,128,8])
    for key in inputdict.keys():
        for ii in range(10):
            data_2d[ii:ii+1,:,:,inputdict[key]] = data[key]
        # data_2d[:,:,:,inputdict[key]] = data[key]

    # grid = get_grid_interp()
    grid=get_grid(GV_RB=True, grid_num=128)
    post = cfdPost_2d(data_2d, grid, inputDict=inputdict)
    # rst = post.get_performance('Static_pressure_ratio', z1=0, z2=127, type='spanwised')
    rst = post.get_performance('Isentropic_efficiency',type='spanwised')
    print(rst)

    plt.plot(rst[0,:], np.linspace(0,1,128))
    plt.show()
