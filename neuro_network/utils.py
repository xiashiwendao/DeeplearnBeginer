import numpy as np
from matplotlib import pyplot as plt


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

# 越阶函数
def step_function(x):
    return np.array(x > 0, dtype=np.int)

def test_sigmoid():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)

    plt.plot(x, y1)
    plt.plot(x, y2, 'k--')
    plt.ylim(-0.1, 1.1) #指定图中绘制的y轴的范围
    plt.show()

def test_relu():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    plt.show()

if __name__ == "__main__":
    # test_relu()
    test_sigmoid()