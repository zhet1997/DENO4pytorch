from flask import Flask, request, jsonify
from set_predictor import build_predicter, run_predicter
import numpy as np

app = Flask(__name__)
model = build_predicter()

@app.route('/predict', methods=['POST'])
def predict():
    # 从请求中获取输入数据
    data = request.json
    input_data = np.array(data['input'])  # 假设输入数据在 'input' 字段中
    outout_para = data['para']
    prediction = run_predicter(input_data, model=model, parameter=outout_para)
    # 返回预测结果
    return jsonify({outout_para: prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
