import os
from post_process.load_model import build_model_yml, loaddata
from model_whole_life import DLModelWhole
from train_task_construct import WorkPrj, add_yml, change_yml, work_construct_togethor


if __name__ == "__main__":
    name = "FNO"
    start_id = 0
    dict = {
    'modes': [4, 6],
    'width': [128],
    'depth': [4, 6],
    'activation': ['gelu', 'relu']
    }

    worklist = work_construct_togethor(dict)

    for id, config_dict in enumerate(worklist):
        work = WorkPrj(os.path.join("..", "work_train_FNO2", name + "_" + str(id)))
        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)
        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(work.device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)


