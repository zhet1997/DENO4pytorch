# running the cfd pred post process
import os
from post_process.model_predict import predictor_establish
# load the model
if __name__ == "__main__":
    name = 'FNO'
    input_dim = 28
    output_dim = 5
    work_load_path = os.path.join("..", 'work', 'model_save')
    model_all = predictor_establish(name, work_load_path)

    parameterList = [
        "Efficiency",
        "EfficiencyPoly",
        "PressureRatioV",
        "TemperatureRatioV",
        "PressureRatioW",
        "TemperatureRatioW",
        "PressureLossR",
        "EntropyStatic",
        "MachIsentropic",
        "Load",
        "MassFlow"
    ]

