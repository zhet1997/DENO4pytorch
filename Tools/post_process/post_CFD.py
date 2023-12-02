import numpy as np
import math

class cfdPost_2d(object):
    def __init__(self,data_2d,grid,inputdict=None): #默认输入格式为64*64*5
        self.grid = grid
        self.data_2d = data_2d

        if inputdict is None:
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

        assert len(self.inputDict.keys())==data_2d.shape[-1], \
            "the physic field name in inputDict is not match with the data_2d"

        self.set_basic_const()
        self.struct_input_check()
        self.get_field_save_dict()
        self.get_performance_save_dict()
        self.get_field_calculate_dict()
        self.get_performance_calculate_dict()

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
        self.hub = self.grid[0,:]
        self.shroud = self.grid[-1, :]


    def struct_input_check(self):
        gridshape = self.grid.shape
        datashape = self.data_2d.shape

        if len(gridshape) != 3 or gridshape[2] != 2:
            print("invalid grid input!")
        if len(gridshape) != 3 and len(gridshape):
            print("invalid data input!")
        if len(datashape) == 3:
            self.data_2d = self.data_2d[None, :, :, :]
            datashape = self.data_2d.shape
            print("one sample input!")
        if len(datashape) == 4:
            self.num = datashape[0]
            print(str(self.num) + " samples input!")
        if gridshape[:2] != datashape[1:3]:
            print("dismatch data & grid input!")

        self.n_1d = self.data_2d.shape[1]
        self.n_2d = self.data_2d.shape[2]

    # def stage_define(self):

    # ===============================================================================#
    # ========================physical field calculation=============================#
    # ===============================================================================#
    def get_field_save_dict(self):
        # all physical field name in dict is same with the Numeca post software.
        self.quanlityList = ['Static Pressure',
                             'Relative Total Pressure',
                             'Absolute Total Pressure',
                             'Rotary Total Pressure',
                             'Static Temperature',
                             'Relative Total Temperature',
                             'Absolute Total Temperature',
                             'Rotary Total Temperature',
                             'Vx', 'Vy', 'Vz','|V|','|V|^2','atan(Vx/Vz)',
                             'Wx', 'Wy', 'Wz','|W|','|W|^2','atan(Wx/Wz)',
                             '|U|',
                             'Speed Of Sound',
                             '|Speed Of Sound|^2',
                             'Relative Mach Number',
                             'Absolute Mach Number',
                             'Static Enthalpy',
                             'Density',
                             'Density Flow',
                             'Entropy',
                             'Static Energy',
                             ]
        self.fieldSaveDict = {}
        self.fieldSaveDict.update({'Gird Node Z': np.tile(self.grid[None, :, :, 0], [self.num, 1, 1])})
        self.fieldSaveDict.update({'Gird Node R': np.tile(self.grid[None, :, :, 1], [self.num, 1, 1])})
        self.fieldSaveDict.update({'R/S Interface': np.where(self.grid[None, :, :, 0]<self.RSInterface, 0, 1)})
        for quanlity in self.quanlityList:
            self.fieldSaveDict.update({quanlity: None})  # initiaize of all fields

    def get_field_calculate_dict(self):
        self.fieldCalculateDict = {}
        self.fieldParaDict = {}
        for quanlity in self.quanlityList:
            self.fieldCalculateDict.update({quanlity: None})
            self.fieldParaDict.update({quanlity: None})

        self.fieldCalculateDict['|U|'] = lambda x1, x2: x1 * x2 * self.rotateSpeed * 2 * np.pi / 60
        self.fieldParaDict['|U|'] = ('Gird Node R','R/S Interface')  # |U| is spectail
        self.fieldCalculateDict['|V|^2'] = lambda x1, x2: (x2 - x1) * 2 * self.Cp
        self.fieldParaDict['|V|^2'] = ('Static Temperature','Absolute Total Temperature')  # |U| is spectail
        self.fieldCalculateDict['|W|^2'] = lambda x1, x2: (x2 - x1) * 2 * self.Cp
        self.fieldParaDict['|W|^2'] = ('Static Temperature', 'Relative Total Temperature')  # |U| is spectail

        self.fieldCalculateDict['Wx'] = lambda x1, x2: x1 + x2
        self.fieldParaDict['Wx'] = ('Vx','|U|')  # |U| is spectail
        self.fieldCalculateDict['Wy'] = lambda x1: x1
        self.fieldParaDict['Wy'] = ('Vy',)  # |U| is spectail
        self.fieldCalculateDict['Wz'] = lambda x1: x1
        self.fieldParaDict['Wz'] = ('Vz',)  # |U| is spectail

        self.fieldCalculateDict['atan(Vx/Vz)'] = lambda x1, x2: np.arctan(x1/x2) / np.pi * 180
        self.fieldParaDict['atan(Vx/Vz)'] = ('Vx', 'Vz')  # |U| is spectail
        self.fieldCalculateDict['atan(Wx/Wz)'] = lambda x1, x2: np.arctan(x1 / x2) / np.pi * 180
        self.fieldParaDict['atan(Wx/Wz)'] = ('Wx', 'Wz')  # |U| is spectail

        self.fieldCalculateDict['Absolute Total Temperature'] = lambda x1, x2: x1 + x2 / 2 / self.Cp #Attention to nonlinear term
        self.fieldParaDict['Absolute Total Temperature']  = ('Static Temperature','|V|^2')
        self.fieldCalculateDict['Relative Total Temperature'] = lambda x1, x2: x1 + x2 / 2 / self.Cp
        self.fieldParaDict['Relative Total Temperature'] = ('Static Temperature', '|W|^2')
        self.fieldCalculateDict['Rotary Total Temperature'] = lambda x1, x2: x1 + np.power(x2, 2) / 2 / self.Cp
        self.fieldParaDict['Rotary Total Temperature'] = ('Static Temperature', '|U|') # |U| is spectail

        self.fieldCalculateDict['Absolute Total Pressure'] = lambda x1, x2, x3: x1 * np.power(x2 / x3, self.kappa / (self.kappa - 1))# Attention to nonlinear term
        self.fieldParaDict['Absolute Total Pressure'] = ('Static Pressure', 'Absolute Total Temperature', 'Static Temperature')
        self.fieldCalculateDict['Relative Total Pressure'] = lambda x1, x2, x3: x1 * np.power(x2 / x3, self.kappa / (self.kappa - 1))
        self.fieldParaDict['Relative Total Pressure'] = ('Static Pressure', 'Relative Total Temperature', 'Static Temperature')
        self.fieldCalculateDict['Rotary Total Pressure'] = lambda x1, x2, x3: x1 * np.power(x2 / x3, self.kappa / (self.kappa - 1))
        self.fieldParaDict['Rotary Total Pressure'] = ('Static Pressure', 'Rotary Total Temperature', 'Static Temperature')

        self.fieldCalculateDict['Speed Of Sound'] = lambda x1, x2: np.sqrt(self.kappa * x1 / x2)
        self.fieldParaDict['Speed Of Sound'] = ('Static Pressure', 'Density')

        self.fieldCalculateDict['|Speed Of Sound|^2'] = lambda x1: (x1) ** 2
        self.fieldParaDict['|Speed Of Sound|^2'] = ('Speed Of Sound',)

        self.fieldCalculateDict['Absolute Mach Number'] = lambda x1, x2: np.sqrt(np.abs(x1) / np.abs(x2))
        self.fieldParaDict['Absolute Mach Number'] = ('|V|^2', '|Speed Of Sound|^2')

        self.fieldCalculateDict['Relative Mach Number'] = lambda x1, x2: np.sqrt(np.abs(x1) / np.abs(x2))
        self.fieldParaDict['Relative Mach Number'] = ('|W|^2', '|Speed Of Sound|^2')

        self.fieldCalculateDict['Static Enthalpy'] = lambda x1:self.Cp* x1
        self.fieldParaDict['Static Enthalpy'] = ('Static Temperature',)

        # self.fieldCalculateDict['Entropy'] = lambda x1, x2, x3: self.Cp*np.log(x1)
        # self.fieldParaDict['Entropy'] = ('Static Temperature', 'Static Pressure', 'Density')

        self.fieldCalculateDict['Absolute Total Enthalpy'] = lambda x1,x2:  x1 + 0.5 * x2
        self.fieldParaDict['Absolute Total Enthalpy'] = ('Static Enthalpy','|V|^2')

        self.fieldCalculateDict['Relative Total Enthalpy'] = lambda x1,x2:  x1 + 0.5 * x2
        self.fieldParaDict['Relative Total Enthalpy'] = ('Static Enthalpy','|W|^2')

        self.fieldCalculateDict['Density Flow'] = lambda x1, x2: x1 * x2
        self.fieldParaDict['Density Flow'] = ('Density', 'Vz')

    def get_field(self,quanlity):
        if self.fieldSaveDict[quanlity] is None: # check whether the field has already calculated
            if quanlity in self.inputDict.keys():# input the field directly
                rst = self.data_2d[..., self.inputDict[quanlity]]
            else: # this field need caluculate with other fields
                rst = self.calculate_field(quanlity)
            self.fieldSaveDict[quanlity] = rst
        else:
            rst = self.fieldSaveDict[quanlity] # return the already exist result
        return rst

    def calculate_field(self, quanlity):
        func = self.fieldCalculateDict[quanlity]
        para = [self.get_field(x) for x in self.fieldParaDict[quanlity]]
        return func(*para)

    def get_mass_weight_radial_averaged_field(self, quanlity, squeeze=True):
        if quanlity in ('Static Pressure', 'Density Flow'):
            rst = self.get_field(quanlity)
        elif quanlity in ('Gird Node R',):
            rst = self.get_field(quanlity)
            rst = np.power(rst[:, -1, None, :], 2) - np.power(rst[:, 0, None, :], 2)  # keep the dim for mean
        else:
            DensityFlow = self.get_field('Density') * self.get_field('Vz')
            rst = DensityFlow * self.get_field(quanlity) / np.mean(DensityFlow, axis=1, keepdims=True)

        rst = np.mean(rst, axis=1, keepdims=True)  # keep the spanwise dim exist

        if squeeze:
            return rst.squeeze(axis=1)
        else:
            return rst

    # def mass_weight_radial_average(self, quanlity, zList=None, nodeList=None): # ,
    #     if quanlity in ('Static Pressure','Density Flow'):
    #         rst = self.get_field(quanlity)
    #     elif quanlity in ('Gird Node R',):
    #         rst = self.get_field(quanlity)
    #         rst = np.power(rst[:,-1, None, :],2) - np.power(rst[:,0, None, :],2) # keep the dim for mean
    #     else:
    #         DensityFlow = self.get_field('Density') * self.get_field('Vz')
    #         rst = DensityFlow * self.get_field(quanlity) / np.mean(DensityFlow, axis=1, keepdims=True)
    #
    #     rst = np.mean(rst, axis=1, keepdims=True) # keep the spanwise dim exist
    #
    #     if nodeList is not None:
    #         assert np.max(nodeList) < self.n_1d
    #         return rst[..., nodeList]
    #     else:
    #         if zList is not None:
    #             idxList = []
    #             weightList = np.zeros([len(zList)])
    #             if not isinstance(zList, list):
    #                 zList = [zList]
    #             for ii, z in enumerate(zList):
    #                 tmp = (self.grid[:,:,0] < z).astype(int)
    #                 idxList[ii] = np.where(tmp==0)[0]
    #                 weightList[ii] = (z - self.grid[idxList[ii],0,0]) / (self.grid[idxList[ii+1],0,0] - self.grid[idxList[ii],0,0])
    #
    #             rst_1 = rst[:, idxList, :, :]
    #             rst_2 = rst[:, [x+1 for x in idxList], :, :]
    #
    #             return weightList*rst_1 + (1-weightList)*rst_2
    #
    #         else:
    #             assert False

    # ===============================================================================#
    # =========================performance calculation===============================#
    # ===============================================================================#
    def get_performance_save_dict(self):
        # all physical field name in dict is same with the Numeca .mf file.
        self.performanceList =  ['Static_pressure_ratio',
                                 'Absolute_total_pressure_ratio',
                                 'Absolute_nozzle_pressure_ratio',
                                 'Relative_nozzle_pressure_ratio',
                                 'Static_temperature_ratio',
                                 'Absolute_total_temperature_ratio',
                                 'Total_total_efficiency',
                                 'Total_static_efficiency',
                                 'Enthalpy',
                                 'Degree_reaction',
                                 'Polytropic_efficiency',
                                 'Isentropic_efficiency',
                                 'Axial_thrust',
                                 'Torque',
                                 'Power',
                                 'Static_Enthalpy',
                                 'Absolute_Enthalpy',
                                 'Relative_Enthalpy',
                                 'Mass_flow',
                                 ]
        self.performanceSaveDict = {}
        for performance in self.performanceList:
            self.performanceSaveDict.update({performance: None})  # initiaize of all fields

    def get_performance_calculate_dict(self): # input are fields in two Z_axis
        self.performanceCalculateDict = {}
        self.performanceParaDict = {}

        for performance in self.performanceList:
            self.performanceCalculateDict.update({performance: None})
            self.performanceParaDict.update({performance: None})

        self.performanceCalculateDict['Static_pressure_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Static_pressure_ratio'] = [('Static Pressure',) for _ in range(2)]

        self.performanceCalculateDict['Absolute_total_pressure_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Absolute_total_pressure_ratio'] = [('Absolute Total Pressure',) for _ in range(2)]

        self.performanceCalculateDict['Absolute_nozzle_pressure_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Absolute_nozzle_pressure_ratio'] = [('Absolute Total Pressure',),
                                                                         ('Static Pressure',)]

        self.performanceCalculateDict['Relative_nozzle_pressure_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Relative_nozzle_pressure_ratio'] = [('Relative Total Pressure',),
                                                                         ('Static Pressure',)]

        self.performanceCalculateDict['Static_temperature_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Static_temperature_ratio'] = [('Static Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Absolute_total_temperature_ratio'] = lambda x1, x2: x1 / x2
        self.performanceParaDict['Absolute_total_temperature_ratio'] = [('Absolute Total Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Static_Enthalpy'] = lambda x1, x2:self.Cp* x1 - self.Cp* x2
        self.performanceParaDict['Static_Enthalpy'] = [('Static Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Absolute_Enthalpy'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        self.performanceParaDict['Absolute_Enthalpy'] = [('Absolute Total Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Relative_Enthalpy'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        self.performanceParaDict['Relative_Enthalpy'] = [('Relative Total Temperature',) for _ in range(2)]

        # self.performanceCalculateDict['Power'] = lambda x1, x2: x1 / x2
        # self.performanceParaDict['Absolute_total_temperature_ratio'] = [('Absolute Total Pressure Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Isentropic_efficiency'] = self._get_Isentropic_efficiency
        self.performanceParaDict['Isentropic_efficiency'] = \
            [('Absolute Total Temperature','Absolute Total Pressure') for _ in range(2)]

        self.performanceCalculateDict['Total_total_efficiency'] = self._get_Total_total_efficiency
        self.performanceParaDict['Total_total_efficiency'] = \
            [('Absolute Total Temperature','Absolute Total Pressure') for _ in range(2)]

        self.performanceCalculateDict['Total_static_efficiency'] = self._get_Total_static_efficiency
        self.performanceParaDict['Total_static_efficiency'] = \
            [('Absolute Total Temperature','Absolute Total Pressure','Static Pressure') for _ in range(2)]

        self.performanceCalculateDict['Degree_reaction'] = self._get_Degree_reaction
        self.performanceParaDict['Degree_reaction'] = \
            [('Absolute Total Pressure','Static Pressure') for _ in range(3)]

        self.performanceCalculateDict['Polytropic_efficiency'] = self._get_Polytropic_efficiency
        self.performanceParaDict['Polytropic_efficiency'] = \
            [('Absolute Total Temperature','Absolute Total Pressure')for _ in range(2)]

        self.performanceCalculateDict['Mass_flow'] =  lambda x1, x2, x3, x4: (x1 * x2 + x3 * x4) * np.pi / 2
        self.performanceParaDict['Mass_flow'] = \
            [('Density Flow','Gird Node R') for _ in range(2)]

    def get_performance(self,performance,
                        type='averaged',
                        z1=None, z2=None):
        # check the up and down stream index
        if z1 is None:
            z1 = 0
        if z2 is None:
            z2 = self.n_2d - 1
        assert z1 < z2

        #select the performance type
        if type=='averaged':
            rst = self.calculate_performance_averaged(performance, z1=z1, z2=z2)
        elif type=='spanwised':
            rst = self.calculate_performance_spanwised(performance, z1=z1, z2=z2)
        elif type=='axiswised':
            rst = self.calculate_performance_averaged(performance, z1=slice([0]*self.n_2d), z2=slice(range(self.n_2d)))
        else:
            assert False

        return rst

    def calculate_performance_averaged(self, performance,
                              z1=None, z2=None, z_middle=71):
        func = self.performanceCalculateDict[performance]
        para = self.performanceParaDict[performance]# it's a tuple contain 2 tuples
        paraValue = []
        computing_station = len(para)
        paraValue = []
        if computing_station == 3:
            zlist = [z1, z_middle, z2]
        else:
            zlist = [z1, z2]

        for ii in range(computing_station):
            for name in para[ii]:
                # get all needed values in the whole axis wise
                if name in self.fieldSaveDict.keys():
                    values = self.get_mass_weight_radial_averaged_field(name) # with 2 dim
                elif name in self.performanceSaveDict.keys():
                    values = self.get_performance(name, type='averaged', z1=z1, z2=z1)
                else:
                    assert False, "the input name is illegal"
                paraValue.append(values[:,zlist[ii]])# get the upstream and downstream point value
        return func(*paraValue)

    def calculate_performance_spanwised(self, performance,
                              z1=None, z2=None, z_middle=71):
        func = self.performanceCalculateDict[performance]
        para = self.performanceParaDict[performance]# it's a tuple contain 2 tuples
        computing_station = len(para)
        paraValue = []
        if computing_station == 3:
            zlist = [z1, z_middle, z2]
        else:
            zlist = [z1, z2]

        for ii in range(computing_station):
            for name in para[ii]:
                # get all needed values in the whole axis wise
                if name in self.fieldSaveDict.keys():
                    values = self.get_field(name)  # with 3 dim
                elif name in self.performanceSaveDict.keys():
                    values = self.get_performance(name, type='spanwised')
                else:
                    assert False, "the input name is illegal"
                paraValue.append(self.get_bar_value(values, zlist[ii]))  # get the upstream and downstream point value

        return func(*paraValue)

    def get_bar_value(self, values, z, bar=2):
        if bar>0:
            return np.mean(values[...,max(z-bar,0):min(z+bar,self.n_2d)], axis=-1)
        else:
            return values[...,z]

    # ===============================================================================#
    # ===========================interface functions=================================#
    # ===============================================================================#

    def get_field_performance(self, name,
                              type='averaged',
                              z1=None, z2=None):
        if z1 is None:
            z1 = 0
        if z2 is None:
            z2 = self.n_2d - 1
        assert z1 < z2

        if name in self.fieldSaveDict.keys():
            if type=='averaged':
                values = self.get_mass_weight_radial_averaged_field(name)
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


    @staticmethod
    def _get_Mach_number():
        print(0)

    def _get_Isentropic_efficiency(self, t1, p1, t2, p2):
        rst1 = (1 - (t2 / t1)) / (1 - np.power(p2 / p1, (self.kappa - 1) / self.kappa))# turbins
        # rst2 = ((np.power(p2 / p1, (self.kappa - 1) / self.kappa) )- 1) / (1 - (t2 / t1))# compressors
        # idx = np.array([1 if x in np.where(p2 < p1)[0].tolist() else 0 for x in range(self.num)])
        # return rst1 * idx + rst2 * (1-idx)
        return rst1
    def _get_Total_total_efficiency(self, t1, p1, t2, p2):
        rst1 = (1 - (t2 / t1)) / (1 - np.power(p2 / p1, (self.kappa - 1) / self.kappa))# turbins
        rst2 = ((np.power(p2 / p1, (self.kappa - 1) / self.kappa) )- 1) / (1 - (t2 / t1))# compressors
        idx = np.array([1 if x in np.where(p2 < p1)[0].tolist() else 0 for x in range(self.num)])
        if len(rst1.shape)==2:
            idx = np.tile(idx[:,np.newaxis], [1, rst1.shape[1]])
        return rst1 * idx + rst2 * (1-idx)

    def _get_Total_static_efficiency(self, t1, tp1, sp1, t2, tp2, sp2):
        rst1 = (1 - (t2 / t1)) / (1 - np.power(sp2 / tp1, (self.kappa - 1) / self.kappa))# turbins
        rst2 = ((np.power(sp2 / tp1, (self.kappa - 1) / self.kappa) )- 1) / (1 - (t2 / t1))# compressors
        idx = np.array([1 if x in np.where(tp2 < tp1)[0].tolist() else 0 for x in range(self.num)])
        if len(rst1.shape)==2:
            idx = np.tile(idx[:,np.newaxis], [1, rst1.shape[1]])
        return rst1 * idx + rst2 * (1-idx)

    def _get_Polytropic_efficiency(self, t1, p1, t2, p2):
        # rst1 = (8.314/self.Cp) / (math.log(p2 / p1)/math.log(t2 / t1))
        rst2 = (self.Cp/8.314) / (np.log(t2 / t1)/np.log(p2 / p1))
        # idx = np.array([1 if x in np.where(p2 < p1)[0].tolist() else 0 for x in range(self.num)])
        # return rst1 * idx + rst2 * (1-idx)
        return rst2

    def _get_Degree_reaction(self, tp1, sp1, tp2, sp2, tp3, sp3):
        rst1 = (sp2 - sp3) / (tp1 - sp3)
        return rst1

if __name__ == "__main__":

    print(0)