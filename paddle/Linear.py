import argparse
from collections import OrderedDict


import paddle
import torch


dont_transpose = [
    "layernorm.weight",
    "_embeddings.weight",
]


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path: str, paddle_dump_path: str):
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        transpose = False
        if k[-7:] == ".weight":
            if not any([w in k for w in dont_transpose]):
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    transpose = True
        print(f"Converting: {k} | is_transpose {transpose}")


        paddle_state_dict[k] = v.data.numpy()


    paddle.save(paddle_state_dict, paddle_dump_path)
