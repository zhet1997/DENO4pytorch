from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from post_process.model_predict import predictor_establish

device = torch.device("cpu")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(model, pytorch_checkpoint_path, save_path=None):
    assert isinstance(save_path, str), "lack of save_path parameter..."
    # create model

    # load model weights
    model_weight_path = pytorch_checkpoint_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # input to the model
    # [batch, channel, height, width]
    # x = torch.rand(1, 3, 224, 224, requires_grad=True, device=device)
    # torch_out = model(x)

    # export the model
    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      save_path,                   # where to save the model (can be a file or file-like object)
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=9,            # the ONNX version to export the model to
                      do_constant_folding=True,    # whether to execute constant folding for optimization
                      input_names=["input"],       # the model's input names
                      output_names=["output"],     # the model's output names
                      )

    # check onnx model
    # onnx_model = onnx.load(save_path)
    # onnx.checker.check_model(onnx_model)
    #
    # ort_session = onnxruntime.InferenceSession(save_path)
    #
    # # compute ONNX Runtime output prediction
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    # ort_outs = ort_session.run(None, ort_inputs)
    #
    # # compare ONNX Runtime and Pytorch results
    # # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")



if __name__ == '__main__':


    # load the model

    if __name__ == "__main__":
        name = 'FNO'
        input_dim = 28
        output_dim = 5
        work_load_path = os.path.join("..", 'work', 'model_save')
        model_all = predictor_establish(name, work_load_path)
