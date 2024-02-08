from Demo.PakB_2d.ablation.ablation_PakB import get_loaders, get_setting, calculate_per
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.Transformers import FourierTransformer
from Demo.PakB_2d.trains_PakB import inference_draw
from Demo.PakB_2d.utilizes_pakB import get_origin
import os
import torch
import numpy as np
os.chdir('E:\WQN\CODE\DENO4pytorch\Demo\PakB_2d/')

if __name__ == "__main__":
    ## data load
    basic_dict, train_dict, pred_model_dict, _ = get_setting()
    dataset_train = [1, 2, 3, 5]
    dataset_valid = [1, 2, 3, 5, 10]
    locals().update(basic_dict)
    locals().update(train_dict)
    name = 'Trans'
    save_number = 0
    work_path = os.path.join('work_ablation', name + '_baseline_' + str(save_number))
    work = WorkPrj(work_path)
    Device = work.device
    train_loader, valid_loader_list, x_normalizer, y_normalizer = get_loaders(dataset_train,
                                                                              dataset_valid,
                                                                              train_num=ntrain,
                                                                              valid_num=nvalid,
                                                                              channel_num=in_dim,
                                                                              batch_size=batch_size,
                                                                              batch_shuffle=False,
                                                                              unpadding=False,
                                                                              )
    _, _, grids = get_origin(type='struct', hole_num=1)
    np.savez(work.grid, grids)
    ## Net_model load
    Net_model = FourierTransformer(**pred_model_dict).to(Device)
    assert os.path.exists(work.pth)
    checkpoint = torch.load(work.pth, map_location=Device)
    Net_model.load_state_dict(checkpoint['net_model'], strict=False)
    Net_model.eval()
    print(0)

    # get pred data
    save_dict = {}
    x, true, pred = inference_draw(train_loader, Net_model, Device,
                                   x_norm=x_normalizer, y_norm=y_normalizer,
                                   shuffle=False,
                                   super_num=1, channel_num=in_dim)
    save_dict.update({'x': x, 'true': true, 'pred': pred})
    np.savez(work.train, **save_dict)

    for ii, loader in enumerate(valid_loader_list):
        save_dict = {}
        x, true, pred = inference_draw(loader, Net_model, Device,
                     x_norm=x_normalizer, y_norm=y_normalizer,
                     shuffle=False,
                     super_num=1, channel_num=in_dim)

        save_dict.update({'x': x, 'true': true, 'pred': pred})
        np.savez(work.valid.replace('.npz','_' + str(dataset_valid[ii]) + '.npz'), **save_dict)




