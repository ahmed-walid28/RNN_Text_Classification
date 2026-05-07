import numpy as np

def activation(x):
    return np.tanh(x)

x1 = 0.05
x2 = 0.10

b1 = 0.5
b2 = 0.7

w1 = np.random.uniform(-0.5, 0.5)
w2 = np.random.uniform(-0.5, 0.5)
w3 = np.random.uniform(-0.5, 0.5)
w4 = np.random.uniform(-0.5, 0.5)

w5 = np.random.uniform(-0.5, 0.5)
w6 = np.random.uniform(-0.5, 0.5)
w7 = np.random.uniform(-0.5, 0.5)
w8 = np.random.uniform(-0.5, 0.5)

net_h1 = x1*w1 + x2*w2 + b1
net_h2 = x1*w3 + x2*w4 + b1

out_h1 = activation(net_h1)
out_h2 = activation(net_h2)

net_o1 = out_h1*w5 + out_h2*w6 + b2
net_o2 = out_h1*w7 + out_h2*w8 + b2

out_o1 = activation(net_o1)
out_o2 = activation(net_o2)

print("Output 1 =", out_o1)
print("Output 2 =", out_o2)