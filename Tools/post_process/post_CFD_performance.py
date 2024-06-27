import numpy as np
class cfdPost_2d(object):
    # ===============================================================================#
    # =========================performance calculation===============================#
    # ===============================================================================#
    def get_performance_save_dict(self):
        # all physical field name in dict is same with the Numeca .mf file.
        self.performanceList = ['Static_pressure_ratio',
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
                                # 'Axial_thrust',
                                # 'Torque',
                                # 'Power',
                                'Static_Enthalpy',
                                'Absolute_Enthalpy',
                                'Relative_Enthalpy',
                                'Mass_flow',
                                'Total_pressure_loss_coefficient',
                                'Total_pressure_recover_coefficient',
                                ]
        self.performanceSaveDict = {}
        for performance in self.performanceList:
            self.performanceSaveDict.update({performance: None})  # initiaize of all fields

    def get_performance_calculate_dict(self):  # input are fields in two Z_axis
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

        # self.performanceCalculateDict['Total_pressure_loss_coefficient'] = lambda p1, u1, rho1, p2, u2, rho2: (p1-p2)/rho1/u1/u1*2
        # self.performanceParaDict['Total_pressure_loss_coefficient'] = [('Absolute Total Pressure','Vz', 'Density') for _ in
        #                                                                 range(2)]

        self.performanceCalculateDict['Total_pressure_loss_coefficient'] = lambda p1, p2: (p1 - p2) / p1
        self.performanceParaDict['Total_pressure_loss_coefficient'] = [('Absolute Total Pressure',) for _ in
                                                                      range(2)]

        # self.performanceCalculateDict['Total_pressure_loss_coefficient'] = lambda pt1, p1, pt2, p2: (pt1 - pt2) / (pt1 - p1)
        # self.performanceParaDict['Total_pressure_loss_coefficient'] = [('Absolute Total Pressure', 'Static Pressure') for _ in
        #                                                                   range(2)]

        self.performanceCalculateDict['Total_pressure_recover_coefficient'] = lambda pt1, pt2, p1, p2: (pt2 - p2) / (pt1 - p2)
        self.performanceParaDict['Total_pressure_recover_coefficient'] = [('Absolute Total Pressure','Static Pressure') for _ in
                                                                       range(2)]

        self.performanceCalculateDict['Static_Enthalpy'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        self.performanceParaDict['Static_Enthalpy'] = [('Static Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Absolute_Enthalpy'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        self.performanceParaDict['Absolute_Enthalpy'] = [('Absolute Total Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Relative_Enthalpy'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        self.performanceParaDict['Relative_Enthalpy'] = [('Relative Total Temperature',) for _ in range(2)]

        # self.performanceCalculateDict['Total_pressure_loss_coefficient'] = lambda x1, x2: self.Cp * x1 - self.Cp * x2
        # self.performanceParaDict['Total_pressure_loss_coefficient'] = [('Absolute Total Pressure',) for _ in range(2)]
        # self.performanceCalculateDict['Power'] = lambda x1, x2: x1 / x2
        # self.performanceParaDict['Absolute_total_temperature_ratio'] = [('Absolute Total Pressure Temperature',) for _ in range(2)]

        self.performanceCalculateDict['Isentropic_efficiency'] = self._get_Isentropic_efficiency
        self.performanceParaDict['Isentropic_efficiency'] = \
            [('Absolute Total Temperature', 'Absolute Total Pressure') for _ in range(2)]

        self.performanceCalculateDict['Total_total_efficiency'] = self._get_Total_total_efficiency
        self.performanceParaDict['Total_total_efficiency'] = \
            [('Absolute Total Temperature', 'Absolute Total Pressure') for _ in range(2)]

        self.performanceCalculateDict['Total_static_efficiency'] = self._get_Total_static_efficiency
        self.performanceParaDict['Total_static_efficiency'] = \
            [('Absolute Total Temperature', 'Absolute Total Pressure', 'Static Pressure') for _ in range(2)]

        self.performanceCalculateDict['Degree_reaction'] = self._get_Degree_reaction
        self.performanceParaDict['Degree_reaction'] = \
            [('Absolute Total Pressure', 'Static Pressure') for _ in range(3)]

        self.performanceCalculateDict['Polytropic_efficiency'] = self._get_Polytropic_efficiency
        self.performanceParaDict['Polytropic_efficiency'] = \
            [('Absolute Total Temperature', 'Absolute Total Pressure') for _ in range(2)]

        self.performanceCalculateDict['Mass_flow'] = lambda x1, x2, x3, x4: (x1 * x2 + x3 * x4) * np.pi / 2
        self.performanceParaDict['Mass_flow'] = \
            [('Density Flow', 'Gird Node R') for _ in range(2)]

    def calculate_performance_averaged(self, performance,
                                       z1=None, z2=None, z_middle=71):
        func = self.performanceCalculateDict[performance]
        para = self.performanceParaDict[performance]  # it's a tuple contain 2 tuples
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
                    values = self.get_mass_weight_radial_averaged_field(name)  # with 2 dim
                elif name in self.performanceSaveDict.keys():
                    values = self.get_performance(name, type='averaged', z1=z1, z2=z1)
                else:
                    assert False, "the input name is illegal"
                paraValue.append(values[:, zlist[ii]])  # get the upstream and downstream point value
        return func(*paraValue)

    def calculate_performance_spanwised(self, performance,
                                        z1=None, z2=None, z_middle=71):
        func = self.performanceCalculateDict[performance]
        para = self.performanceParaDict[performance]  # it's a tuple contain 2 tuples
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
        if bar > 0:
            return np.mean(values[..., max(z - bar, 0):min(z + bar, self.n_2d)], axis=-1)
        else:
            return values[..., z]


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
        # rst2 = ((np.power(p2 / p1, (self.kappa - 1) / self.kappa) )- 1) / (1 - (t2 / t1))# compressors
        # idx = np.array([1 if x in np.where(p2 < p1)[0].tolist() else 0 for x in range(self.num)])
        # if len(rst1.shape)==2:
        #     idx = np.tile(idx[:,np.newaxis], [1, rst1.shape[1]])
        # return rst1 * idx + rst2 * (1-idx)
        return rst1

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

