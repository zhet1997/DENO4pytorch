import os
import paddle
from post_process.load_model import build_model_yml, loaddata
from model_whole_life import DLModelWhole
from train_task_construct import WorkPrj, add_yml, change_yml, work_construct

if __name__ == "__main__":
    name = "Transformer"
    start_id = 0

    dict = {
        'ntrain': [100, 500, 1000, 1500, 2000],
        # 'noise_scale': [0.005, 0.01, 0.05, 0.1],
    }
    worklist = work_construct(dict)

    for id, config_dict in enumerate(worklist):
        work = WorkPrj(os.path.join("..", "work_noise_Trans1", name + "_" + str(id + start_id)))
        change_yml("Basic", yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", name+"_config"], yml_path=work.yml)
        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(work.device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)


