import os
from post_process.load_model import build_model_yml, loaddata
from model_whole_life import DLModelWhole
from train_task_construct import WorkPrj, add_yml, change_yml, work_construct_togethor


if __name__ == "__main__":
    name = "TNO"
    start_id = 0

    dict_model = {
    }

    model_list = work_construct_togethor(dict_model)

    for id, config_dict in enumerate(model_list[start_id:]):
        work = WorkPrj(os.path.join("..", "work_trainsql_Trans2", name + "_" + str(id)))

        # work = WorkPrj(os.path.join("..", "work_train_Trans4", name + "_" + str(id+ start_id)))

        change_yml(name, yml_path=work.yml, **config_dict)
        add_yml(["Optimizer_config", "Scheduler_config", "Basic_config"], yml_path=work.yml)

        train_loader, valid_loader, x_normalizer, y_normalizer = loaddata(name, **work.config("Basic"))
        x_normalizer.save(work.x_norm)
        y_normalizer.save(work.y_norm)
        DL_model = DLModelWhole(work.device, name=name, work=work)
        DL_model.set_los()
        DL_model.train_epochs(train_loader, valid_loader)
