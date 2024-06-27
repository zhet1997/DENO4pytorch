import os
import matplotlib.pyplot as plt
import numpy as np
from Demo.GVRB_2d.utilizes_GVRB import get_grid_interp
from Tools.post_process.model_predict import DLModelPost
from Tools.uncertainty.GVRB_setting import InputTransformer, UQTransformer
from Utilizes.visual_data import MatplotlibVision
from Tools.optimization.pymoo_optimizer import TurboPredictor, predictor_establish
# def dominates(row, candidateRow):
#     return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)
#
# def simple_cull(inputPoints):
#     paretoPoints = []
#     candidateRowNr = 0
#     dominatedPoints = []
#     while True:
#         candidateRow = inputPoints[candidateRowNr]
#         inputPoints.remove(candidateRow)
#         rowNr = 0
#         nonDominated = True
#         while len(inputPoints) != 0 and rowNr < len(inputPoints):
#             row = inputPoints[rowNr]
#             if dominates(candidateRow, row):
#                 # If it is worse on all features remove the row from the array
#                 inputPoints.remove(row)
#                 dominatedPoints.append(list(row))
#             elif dominates(row, candidateRow):
#                 nonDominated = False
#                 dominatedPoints.append(list(candidateRow))
#                 rowNr += 1
#             else:
#                 rowNr += 1
#
#         if nonDominated:
#             # add the non-dominated point to the Pareto frontier
#             paretoPoints.append(list(candidateRow))
#
#         if len(inputPoints) == 0:
#             break
#
#     return np.array(paretoPoints), np.array(dominatedPoints)

def dominates(row, candidateRow):
    return all(row <= candidateRow) and any(row < candidateRow)

def simple_cull(inputPoints):
    paretoPoints = []
    paretoIdx = []
    candidateRowNr = 0
    dominatedPoints = []
    inputIdx = list(range(len(inputPoints)))  # 创建索引列表
    while True:
        candidateRow = inputPoints[candidateRowNr]
        candidateIdx = inputIdx[candidateRowNr]  # 获取候选点的索引
        inputPoints = np.delete(inputPoints, candidateRowNr, axis=0)
        inputIdx.pop(candidateRowNr)  # 从索引列表中删除候选点索引
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                inputPoints = np.delete(inputPoints, rowNr, axis=0)
                dominatedPoints.append(list(row))
                inputIdx.pop(rowNr)  # 从索引列表中删除支配点索引
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.append(list(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            paretoPoints.append(list(candidateRow))
            paretoIdx.append(candidateIdx)  # 记录非支配点的索引

        if len(inputPoints) == 0:
            break

    return np.array(paretoPoints), np.array(dominatedPoints), np.array(paretoIdx)

def draw_bc_curve(Visual, rst, label=None, xlim=None, ylim=None, fig=None, axs=None):
    # colorList = ['steelblue', 'darkslateblue']
    # index = [0, 1, 2, 9, 3, 4]

    # colors = np.take(cmap.colcen']
    shape = rst.shape
    markerList = ['-'] * shape[0]
    # labelList = ['A'] * shape[0]
    labelList = label
    # colorList = ['r'] * shape[0]
    # colorList = [None] * shape[0]
    cmap = plt.cm.get_cmap('tab10')
    colorList = list(cmap.colors[:3])
    if rst.shape[0] >3:
        for _ in range(rst.shape[0]-3):
            colorList.append(colorList[2])
            labelList.append(None)


    Visual.plot_curve_scatter(fig, axs,
                              x=np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)),
                              y=rst,
                              labelList=labelList,
                              colorList=colorList, markerList=markerList, xlim=xlim, ylim=ylim,
                              xylabels=('value', "span"))

def draw_performance_with_bc_change(X, var_name=None, uq_number = 100, work_load_path=None,
                   save_path=None, parameterList=None, uqList=None,
                                    ):
    ## draw the figure
    model_all = predictor_establish('TNO', work_load_path)
    Visual = MatplotlibVision(' ', input_name=('x', 'y'), field_name=('none'))
    Visual.font_EN = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    Visual.font_CHN = {'family': 'SimSun', 'weight': 'normal', 'size': 10}

    for uq_name in uqList:
        adapter_gvrb = UQTransformer(var_name, uq_name=uq_name, uq_number=uq_number)
        P = optiaml_predicter(model=model_all,
                              adapter=adapter_gvrb,
                              parameterList=parameterList,
                              )
        F_bc = P.evaluate_with_bc_change(X, perturb=False)
        F_bc = F_bc.reshape([uq_number,X.shape[0],len(parameterList)]).transpose(1,0,2)
        for ii, parameter in enumerate(parameterList):
            fig, axs = plt.subplots(1, 1, figsize=(7, 9))
            plt.cla()
            draw_bc_curve(Visual, F_bc[...,ii], label=['baseline', 'optimal_dtm', 'optimal_uq'], fig=fig, axs=axs, xlim=[0,1])
            fig.savefig(os.path.join(save_path, uq_name + '_' +parameter.replace('/','') + '_compare' +'.jpg'))

class optiaml_predicter(object):
    def __init__(self,
                 model: DLModelPost=None,
                 adapter: InputTransformer=None,
                 parameterList=None,
                ):

        self.model = model
        self.adapter = adapter
        self.parameterList = parameterList

    def evaluate_with_bc_change(self, x, type='linear',space=0, perturb=True):
        if len(x.shape)==1:
            x = x[np.newaxis,:]
        grid = get_grid_interp()
        rst = self.model.predictor_cfd_value(
                self.adapter.input_transformer(x, type=type, perturb=perturb),
                parameterList=self.parameterList,
                setOpt=False,
                space=space,
                grid=grid,
            )
        return rst


if __name__ == "__main__":
    save_path = 'E:\WQN\CODE\DENO4pytorch\Demo\GVRB_2d\work_opt\opt_rst_2/'
    file_name = os.path.join(save_path, 'uq_opt.npz')
    data = np.load(file_name)
    print(0)
    F = data['F'].reshape([-1, data['F'].shape[-1]])
    paretoPoints, dominatedPoints = simple_cull(F.tolist())
    plt.scatter(dominatedPoints[:, 0], dominatedPoints[:, 1], s=10, alpha=0.5)
    plt.scatter(paretoPoints[:,0], paretoPoints[:,1], s=20, alpha=0.5)
    # plt.savefig(save_path + 'pareto.png', dpi=600)
    plt.show()

    print(0)

