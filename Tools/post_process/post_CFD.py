import numpy as np
import torch
from Tools.post_process.post_CFD_similarity import cfdPost_2d as similarity
from Tools.post_process.post_CFD_field import cfdPost_2d as field
from Tools.post_process.post_CFD_performance import cfdPost_2d as performance

def apply_normalizer_similarity(func):
    def wrapper(instance, data_raw, dimensional_matrix, vtype=None):
        # Apply normalization if necessary
        if vtype == 'x':
            if instance.x_norm is not None:
                data_raw = instance.x_norm.back(data_raw)
            data_raw = func(instance, data_raw, dimensional_matrix, vtype)
            if instance.x_norm is not None:
                data_raw = instance.x_norm.norm(data_raw)
        elif vtype == 'y':
            if instance.y_norm is not None:
                data_raw = instance.y_norm.back(data_raw)
            data_raw = func(instance, data_raw, dimensional_matrix, vtype)
            if instance.y_norm is not None:
                data_raw = instance.y_norm.norm(data_raw)
        else:
            data_raw = func(instance, data_raw, dimensional_matrix, vtype)

        return data_raw

    return wrapper


def apply_readin_without_grid(func):
    def wrapper(instance, *args, **kwargs):
        if instance.grid is not None:
            func(instance, *args, **kwargs)
        else:
            instance.num_raw = instance.data_raw.shape[0]
            instance.n_1d = instance.data_raw.shape[1]
            instance.n_2d = instance.data_raw.shape[2]
    return wrapper


def apply_normalizer_readin(func):
    def wrapper(instance, *args, **kwargs):
        if 'boundarycondition' in kwargs.keys():
            if torch.is_tensor(kwargs['boundarycondition']):
                kwargs['boundarycondition'] = kwargs['boundarycondition'].cpu().numpy()
            # if len(kwargs['boundarycondition'].shape)>2:
            #     kwargs['boundarycondition'] = kwargs['boundarycondition'][:,0,0,-4:]
        if 'data' in kwargs.keys():
            if torch.is_tensor(kwargs['data']):
                kwargs['data'] = kwargs['data'].cpu().numpy()

        if 'x_norm' in kwargs.keys():
            instance.x_norm=kwargs['x_norm']
            kwargs['boundarycondition'] = instance.x_norm.back(kwargs['boundarycondition'])
            kwargs.pop('x_norm')
        else:
            instance.x_norm = None
        if 'y_norm' in kwargs.keys():
            instance.y_norm=kwargs['y_norm']
            kwargs['data']=instance.y_norm.back(kwargs['data'])
            kwargs.pop('y_norm')
        else:
            instance.y_norm = None
        func(instance, *args, **kwargs)

    return wrapper

def apply_normalizer_export(func):
    def wrapper(instance, *args, **kwargs):
        if instance.x_norm is not None:
            instance.bouCondition_1d = torch.tensor(instance.x_norm.norm(instance.bouCondition_1d), dtype=torch.float)
        if instance.y_norm is not None:
            instance.data_2d = torch.tensor(instance.y_norm.norm(instance.data_2d), dtype=torch.float)
        loader = func(instance, *args, **kwargs)

        return loader
    return wrapper

class cfdPost_2d(similarity, field, performance):
    def __init__(self, **kwargs): #默认输入格式为64*64*5
        self.par_init = False
        self.sim_init = False
        if len(kwargs)>0:
            self.field_data_readin(**kwargs)

    def paramter_calculate_init(self):
        self.par_init = True
        self.get_field_save_dict()
        self.get_performance_save_dict()
        self.get_field_calculate_dict()
        self.get_performance_calculate_dict()

    def similarity_calculate_init(self):
        self.sim_init = True
        self.get_dimensional_save_dict()

    #===============================================================================#
    #==========================input and output part================================#
    #===============================================================================#

    @apply_normalizer_readin
    def field_data_readin(self,data=None,grid=None,
                          inputdict={},
                          boundarydict={},
                          boundarycondition=None,
                          similarity=False):
        # if grid is not None:
        self.grid = grid
        # if data is not None:
        self.data_raw = data
        # if boundarycondition is not None:
        self.bouCondition_raw = boundarycondition
        if inputdict=={}:
            self.inputDict = {'Static Pressure': 0,
                              'Static Temperature': 1,
                              'Density': 2,
                              'Vx': 3,
                              'Vy': 4,
                              'Vz': 5,
                              'Relative Total Temperature': 6,
                              'Absolute Total Temperature': 7,
                              }
        else:
            self.inputDict = inputdict

        if boundarycondition is not None and boundarydict=={}:
            self.boundaryDict = {'Absolute Total Pressure': 0,
                              'Absolute Total Temperature': 1,
                              'Absolute Static Pressure': 2,
                              'Rotational_speed': 3,
                              }
        else:
            self.boundaryDict = boundarydict

        assert len(self.inputDict.keys()) == data.shape[-1], \
            "the physic field name in inputDict is not match with the data_2d"

        self.set_basic_const()
        self.struct_input_check()

        if not similarity:
            self.data_2d = self.data_raw
            self.bouCondition_1d = self.bouCondition_raw
            self.num = self.num_raw

    @apply_normalizer_readin
    def bouCondition_data_readin(self,
                                  inputdict={},
                                  boundarydict={},
                                  boundarycondition=None,
                                 ):
        self.bouCondition_raw = boundarycondition
        if inputdict == {}:
            self.inputDict = {'Static Pressure': 0,
                              'Static Temperature': 1,
                              'Density': 2,
                              'Vx': 3,
                              'Vy': 4,
                              'Vz': 5,
                              'Relative Total Temperature': 6,
                              'Absolute Total Temperature': 7,
                              }
        else:
            self.inputDict = inputdict

        if boundarycondition is not None and boundarydict == {}:
            self.boundaryDict = {'Flow Angle': 0,
                                 'Absolute Total Temperature': 1,
                                 'Absolute Total Pressure': 2,
                                 'Rotational_speed': 3,
                                 }
        else:
            self.boundaryDict = boundarydict

        self.num_raw = self.bouCondition_raw.shape[0]

        self.set_basic_const()

    def loader_readin(self, loader, **kwargs):
        x, y, batchsize = self.recover_data_from_loader(loader)
        self.loader = {}
        if len(x.shape)>2:
            x = x[:,0,0,:]
            self.loader.update({'expand': True})
        self.field_data_readin(data=y,
                               boundarycondition=x,#[:,-4:],#specially
                               **kwargs,
                               )
        self.loader.update({'x': x})
        self.loader.update({'batch_size': batchsize})


    @apply_normalizer_export
    def loader_export(self, expand=1):
        x = self.loader['x'].repeat_interleave(expand, dim=0) #here has a mistake while expand > 0!!!!!!!!
        x = torch.as_tensor(self.bouCondition_1d, dtype=torch.float).detach().clone()
        # x[:,-4:] = torch.as_tensor(self.bouCondition_1d, dtype=torch.float).detach().clone()
        if 'expand' in self.loader.keys():
            x = x.unsqueeze(1).unsqueeze(2).expand(-1, 64, 128, -1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, torch.as_tensor(self.data_2d, dtype=torch.float).detach().clone()),
                                                   batch_size=self.loader['batch_size'], shuffle=False, drop_last=False)

        return loader

    def loader_similarity(self, loader, grid=None, scale=[-0.1,0.1], expand=1, log=False, **kwargs):
        self.loader_readin(loader, grid=grid, similarity=True, **kwargs)
        self.get_data_after_similarity(expand=expand, scale=scale, log=log)
        return self.loader_export(expand=expand)

    @staticmethod
    def recover_data_from_loader(data_loader):
        x_list = []
        y_list = []
        batch_size=None

        # 遍历数据加载器的迭代器
        for batch_x, batch_y in data_loader:
            x_list.append(batch_x)
            y_list.append(batch_y)
            if batch_size is None:
                batch_size = batch_x.size(0)

        # 拼接所有 batch 中的输入和标签
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)

        return x, y, batch_size


    #===============================================================================#
    #==========================parameter setting part===============================#
    #===============================================================================#
    def set_basic_const(self,
                        kappa = 1.400,
                        Cp = 1004,
                        sigma = 1.6,
                        rotateSpeed = 8279, # rpm
                        Rg = 287,
                        RSInterface = 0.042
                        ):
        self.kappa = float(kappa)
        self.Cp = float(Cp)
        self.sigma = float(sigma)# 代表稠度
        self.rotateSpeed = float(rotateSpeed)
        self.Rg = float(Rg)
        self.RSInterface = float(RSInterface)

    def get_hub_shroud(self):
        self.hub = self.grid[0, :]
        self.shroud = self.grid[-1, :]

    def set_change_rotateSpeed(self):
        rotateSpeed = self.bouCondition_1d[:, self.boundaryDict['Rotational Speed']]
        self.fieldSaveDict.update({'Rotational Speed': np.tile(rotateSpeed[:, None, None, :], [1, self.n_1d, self.n_2d])})

    @apply_readin_without_grid
    def struct_input_check(self):
        gridshape = self.grid.shape
        datashape = self.data_raw.shape

        if len(gridshape) != 3 or gridshape[2] != 2:
            print("invalid grid input!")
        if len(gridshape) != 3 and len(gridshape):
            print("invalid data input!")
        if len(datashape) == 3:
            self.data_raw = self.data_raw[None, :, :, :]
            datashape = self.data_raw.shape
            self.num_raw = 1
            print("one sample input!")
        if len(datashape) == 4:
            self.num_raw = datashape[0]
            print(str(self.num_raw) + " samples input!")
        if gridshape[:2] != datashape[1:3]:
            print("dismatch data & grid input!")

        self.n_1d = self.data_raw.shape[1]
        self.n_2d = self.data_raw.shape[2]

    # def stage_define(self):





    # ===============================================================================#
    # ===========================interface functions=================================#
    # ===============================================================================#

    def get_field_performance(self, name,
                              type='averaged',
                              z1=None, z2=None):
        if not self.par_init:
            self.paramter_calculate_init()

        if z1 is None:
            z1 = 0
        if z2 is None:
            z2 = self.n_2d - 1
        assert z1 <= z2

        if name in self.fieldSaveDict.keys():
            if type=='averaged':
                values = self.get_mass_weight_radial_averaged_field(name)
                values = values.mean(axis=-1)
            elif type=='spanwised':
                values = self.get_field(name)  # with 3 dim
                values = self.get_bar_value(values, z2)
        elif name in self.performanceSaveDict.keys():
            values = self.get_performance(name, type=type, z1=z1, z2=z2)
        else:
            assert False
        return values

    def get_fields(self, quanlityList):
        if not isinstance(quanlityList,list):
            quanlityList = [quanlityList]
        rst = np.zeros([self.num, self.n_1d, self.n_2d, len(quanlityList)])
        for ii, quanlity in enumerate(quanlityList):
            rst[..., ii] = self.get_field(quanlity)
        return rst

    def get_performance(self, performance,
                        type='averaged',
                        z1=None, z2=None):
        if not self.par_init:
            self.paramter_calculate_init()

        # check the up and down stream index
        if z1 is None:
            z1 = 0
        if z2 is None:
            z2 = self.n_2d - 1
        assert z1 < z2

        # select the performance type
        if type == 'averaged':
            rst = self.calculate_performance_averaged(performance, z1=z1, z2=z2, z_middle=int((z1+z2)/2))
        elif type == 'spanwised':
            rst = self.calculate_performance_spanwised(performance, z1=z1, z2=z2)
        elif type == 'axiswised':
            rst = self.calculate_performance_averaged(performance, z1=slice([0] * self.n_2d),
                                                      z2=slice(range(self.n_2d)))
        else:
            assert False

        return rst


if __name__ == "__main__":

    print(0)