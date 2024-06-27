import onnx
import os


if __name__ == "__main__":
    os.chdir(r'E:\\WQN\\CODE\\DENO4pytorch/')
    name = 'deepONet'
    work_load_path = os.path.join('Demo', 'GVRB_2d', 'work_opt')
    onnx_model = onnx.load(os.path.join(work_load_path, "don_test.onnx"))
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print("Model incorrect")
    else:
        print("Model correct")