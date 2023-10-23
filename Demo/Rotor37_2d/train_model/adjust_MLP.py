import os
from post_process.load_model import build_model_yml, loaddata
from model_whole_life import DLModelWhole
from train_task_construct import WorkPrj, add_yml, change_yml, work_construct

if __name__ == "__main__":
    name = "MLP"
    start_id = 0
    dict_model = {
                "n_hidden": [256, 512],
                "num_layers": [8, 12, 14],
                "is_BatchNorm": [True, False]
                }

    model_list = work_construct(dict_model)

    for id, config_dict in enumerate(model_list):
        work = WorkPrj(os.path.join("..", "work", name + "_" + str(id + start_id)))

        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)

        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(work.device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader, save_iter=5)
