import os
import scipy.io as scio
from scipy.interpolate import griddata, interp1d
from Demo.GVRB_2d.utilizes_GVRB import get_unstruct_quanlity_from_mat
import numpy as np

def interpolate_2d(points, values, points_new, method='linear'):

    Z_interpolated = griddata(points, values, points_new, method='linear')
    # func = Rbf(points[:,0], points[:,1], values, function='linear')  # 插值
    # Z_interpolated = func(points_new[:,0],points_new[:,1])
    return Z_interpolated


def get_grid_interp(grid_num_s=128,
                    grid_num_z=128,
                    z_inlet=-0.059,
                    z_outlet=0.119,
                    hub_adjust=0.0001,
                    shroud_adjust=0.0001,
                    data_path=None,
                    ):
    if data_path is not None:
        shroud_file = os.path.join(data_path,"shroud.dat")
        hub_file = os.path.join(data_path,"hub.dat")
    else:
        shroud_file = os.path.join('data', "shroud.dat")
        hub_file = os.path.join('data', "hub.dat")
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
        yy[i][0] = yy[i][0]+0.00005

    yy = np.concatenate(yy, axis=0)
    yy = yy.reshape(grid_num_z, grid_num_s).T
    xx = xx.reshape(grid_num_s, grid_num_z)

    return np.concatenate([xx[:,:,np.newaxis],yy[:,:,np.newaxis]],axis=2)

def readAllfield(quanlityList=None):

    sample_files = [os.path.join("data", "sampleRstUnstruct_3100"),
                    ]

    grid, fields, invalid_idx = get_unstruct_quanlity_from_mat(sample_files, quanlityList=quanlityList,
                                                               invalid_idx=True)

    dict = {}
    for ii, quanlity in enumerate(quanlityList):
        dict.update({quanlity : fields[:,:5500,ii]})

    return grid[:,:5500,:], dict, invalid_idx

if __name__ == "__main__":
    quanlityList = ["Static Pressure", "Static Temperature", "Density",
                    "Vx", "Vy", "Vz",
                    'Relative Total Temperature',
                    'Absolute Total Temperature']

    data_path = os.path.join('data')
    grid = get_grid_interp(data_path=data_path, grid_num_s=64, grid_num_z=128)
    print(grid.shape)
    field_dict_new = {}
    field_dict_new.update({'grid': grid})
    grid = grid.reshape([-1, 2])
    grid_unstruct, field_dict_unstruct, invalid_idx= readAllfield(quanlityList=quanlityList)
    field_dict_new.update({'invalid_idx': invalid_idx})
    sampleNum = grid_unstruct.shape[0]

    for quanlity in quanlityList:
        value_new = np.zeros([sampleNum, grid.shape[0]])
        for ii in range(sampleNum):
            if ii not in invalid_idx:
                temp= interpolate_2d(grid_unstruct[ii,...], field_dict_unstruct[quanlity][ii,...], grid)
                isnan = np.isnan(temp)
                if True in isnan:
                    temp2 = interpolate_2d(grid_unstruct[ii,...], field_dict_unstruct[quanlity][ii,...], grid, method='nearest')
                    temp[isnan] = temp2[isnan]
                value_new[ii, ...] = temp
            print(ii)
        field_dict_new.update({quanlity : value_new.copy().reshape([sampleNum,64,128])})
    scio.savemat(os.path.join(data_path, "sampleStruct_fixGemo_128_64_" + str(sampleNum) + ".mat"),
                 field_dict_new)


