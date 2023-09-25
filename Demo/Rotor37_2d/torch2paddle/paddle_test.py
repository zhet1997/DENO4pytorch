import paddle
import paddle.nn as nn
import random
import matplotlib.pyplot as plt
import numpy as np
# 生成随机数据
np.random.seed(2021)
x = np.random.rand(100, 1).astype('float32') * 5
y = x * 3.0 + 2.0 + np.random.randn(100, 1).astype('float32') * 0.1

# 将数据转换为Paddle Tensor格式
x = paddle.to_tensor(x)
y = paddle.to_tensor(y)


# 定义模型
class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.fc(x)


model = LinearNet()

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=0.01)

# 训练模型
epochs = 500
for epoch in range(epochs):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    if epoch % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.numpy()[0]:.4f}')

# 绘制结果
y_pred = model(x)
plt.plot(x.numpy(), y.numpy(), 'o', label='data')
plt.plot(x.numpy(), y_pred.numpy(), label='fitting line')
plt.legend()
plt.show()
