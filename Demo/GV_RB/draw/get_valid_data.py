import yaml
from post_process.load_model import loaddata_Sql
from model_whole_life import WorkPrj, DLModelWhole, change_yml, add_yml
from TestData.post_CFD import cfdPost_2d
from utilizes_rotor37 import get_grid, get_origin_GVRB