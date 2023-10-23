import argparse
from collections import OrderedDict
import paddle
import torch
import os
import numpy as np

dont_transpose = [
    "layernorm.weight",
    "_embeddings.weight",
]
def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path: str, paddle_dump_path: str):
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    # paddle_state_dict = dict()
    for k, v in pytorch_state_dict['net_model'].items():
        transpose = False
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1) #why??
                    transpose = True
        print(f"Converting: {k} | is_transpose {transpose}")
        # if not k=='log_loss':
        temp =  v.data.numpy()
        paddle_state_dict[k] = paddle.to_tensor(temp,  dtype='float32')

    paddle.save(paddle_state_dict, paddle_dump_path)
    # print(paddle_state_dict)

if __name__ == "__main__":
    path_model =  r"D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\work\model_save\TNO/"
    pytorch_checkpoint_path = os.path.join(path_model, 'latest_model.pth')
    paddle_dump_path = os.path.join(path_model, 'latest_model.pdparams')
    path_pd = "D:\WQN\CODE\DENO4pytorch-main\Demo\Rotor37_2d\work\MLP_0\latest_model.pdparams"



    convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path, paddle_dump_path)
    # check_point_1 = torch.load(pytorch_checkpoint_path, map_location='cpu')['net_model']
    # check_point_2 = paddle.load(paddle_dump_path)
    # check_point_3 = paddle.load(path_pd)

    print(0)