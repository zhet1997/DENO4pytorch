import numpy as np
class cfdPost_2d(object):
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
                             'Rotational Speed',
                             ]
        self.fieldSaveDict = {}
        for quanlity in self.quanlityList:
            self.fieldSaveDict.update({quanlity: None})  # initiaize of all fields
        self.fieldSaveDict.update({'Gird Node Z': np.tile(self.grid[None, :, :, 0], [self.num, 1, 1])})
        self.fieldSaveDict.update({'Gird Node R': np.tile(self.grid[None, :, :, 1], [self.num, 1, 1])})
        self.fieldSaveDict.update({'R/S Interface': np.where(self.grid[None, :, :, 0]<self.RSInterface, 0, 1)})

        if 'Rotational Speed' in self.boundaryDict.keys():
            self.set_change_rotateSpeed()
        else:
            self.fieldSaveDict.update({'Rotational Speed': np.tile(np.array(self.rotateSpeed)[None, None, None],
                                                                   [self.num, self.n_1d, self.n_2d])})



    def get_field_calculate_dict(self):
        self.fieldCalculateDict = {}
        self.fieldParaDict = {}
        for quanlity in self.quanlityList:
            self.fieldCalculateDict.update({quanlity: None})
            self.fieldParaDict.update({quanlity: None})

        self.fieldCalculateDict['|U|'] = lambda x1, x2, x3: x1 * x2 * x3 * 2 * np.pi / 60
        self.fieldParaDict['|U|'] = ('Gird Node R','R/S Interface', 'Rotational Speed')  # |U| is spectail
        self.fieldCalculateDict['|V|^2'] = lambda x1, x2: (x2 - x1) * 2 * self.Cp
        self.fieldParaDict['|V|^2'] = ('Static Temperature','Absolute Total Temperature')  # |U| is spectail
        self.fieldCalculateDict['|W|^2'] = lambda x1, x2: (x2 - x1) * 2 * self.Cp
        self.fieldParaDict['|W|^2'] = ('Static Temperature', 'Relative Total Temperature')  # |U| is spectail

        self.fieldCalculateDict['|V|'] = lambda x1, x2, x3: np.power(x1*x1 + x2*x2 + x3*x3,0.5)
        self.fieldParaDict['|V|'] = ('Vx', 'Vy', 'Vz')  # |U| is spectail
        self.fieldCalculateDict['|W|'] = lambda x1, x2, x3: np.power(x1 * x1 + x2 * x2 + x3 * x3, 0.5)
        self.fieldParaDict['|W|'] = ('Wx', 'Wy', 'Wz')  # |U| is spectail


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

        self.fieldCalculateDict['Absolute Total Temperature'] = lambda x1, x2: x1 + x2*x2 / 2 / self.Cp #Attention to nonlinear term
        self.fieldParaDict['Absolute Total Temperature']  = ('Static Temperature','|V|')
        self.fieldCalculateDict['Relative Total Temperature'] = lambda x1, x2: x1 + x2*x2 / 2 / self.Cp
        self.fieldParaDict['Relative Total Temperature'] = ('Static Temperature', '|W|')
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
        if not self.par_init:
            self.paramter_calculate_init()
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