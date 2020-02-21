# coding: utf-8
import sys, os
sys.path.append(os.getcwd())  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
# 注意这里是关键：
# 1. 在深度学习里面，W就是要求的梯度的对象，扮演的是X的角色
# 因为在模型训练中，样本X, y是已知的，但是参数W是未知的；
# 2. 这里的求梯度的函数就是损失函数，求得是损失函数偏W的导数
dW = numerical_gradient(f, net.W)

print(dW)
