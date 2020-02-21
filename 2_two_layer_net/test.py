import sys, os
sys.path.append(os.getcwd())  # 为了导入父目录的文件而进行的设定
from two_layer_net import TwoLayerNet

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
print(network.params['W1'])
