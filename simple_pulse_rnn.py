#!/usr/bin/env python

import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

class ImpulseData(object): #TODO remove object
    """
    x - network inputs
    y - desired output
    """
    def __init__(self, data_len):
        self.x = np.zeros((data_len, 2), np.float32)
        self.y = np.zeros((data_len, 1), np.float32)
        period = 11
        i = 0
        while i < data_len:
            a = np.random.randint(2)
            self.x[i, 0] = a
            if a == 1:
                b = np.random.rand()
                self.x[i, 1] = b
                pulse_len = int(b * 10)
                if i + pulse_len < data_len:
                    self.y[i:i + pulse_len, 0] = 1
                else:
                    self.y[i:-1, 0] = 1
                i += period
            i += 1

train_data_len = 200000
train_data = ImpulseData(train_data_len)
X = train_data.x
y = train_data.y

valid_data_len = 200
valid_data = ImpulseData(valid_data_len)
Xv = valid_data.x
yv = valid_data.y

def activation(x):
    output = 1/(1 + np.exp(-x))
    return output

def activation_derivative(output):
    return output * (1 - output)

nu = 0.001
input_dim = 2
hidden_dim = 180
output_dim = 1

W_0 = np.random.uniform(low=-0.9, high=0.9, size=(hidden_dim, input_dim))
W_1 = np.random.uniform(low=-0.9, high=0.9, size=(output_dim, hidden_dim))
W_h = np.random.uniform(low=-0.9, high=0.9, size=(hidden_dim, hidden_dim))

d_0_l = np.zeros_like(W_0)
d_1_l = np.zeros_like(W_1)
d_h_l = np.zeros_like(W_h)

backprop_depth = 20
inputs_unrolled = [np.zeros((input_dim, 1), np.float32)] * backprop_depth
h_unrolled = [np.zeros((hidden_dim, 1), np.float32)] * backprop_depth
d_0_unrolled = [np.zeros((hidden_dim, 1), np.float32)] * backprop_depth
d_1_unrolled = [np.zeros((output_dim, 1), np.float32)] * backprop_depth

def forvard_pass(x, W_0, W_1, W_h, h_prev):
    v_0 = np.dot(W_0, x)
    v_h = np.dot(W_h, h_prev)
    h = activation(v_0 + v_h)
    v1 = np.dot(W_1, h)
    y = activation(v1)

    return y, h

for j in range(train_data_len):
    sample = X[j, :].reshape((input_dim, 1))
    t = y[j, 0].reshape((output_dim, 1))
    h_prev = h_unrolled[-1]
    y_1, h = forvard_pass(sample, W_0, W_1, W_h, h_prev)

    # backward pass
    e = t - y_1
    d_1 = e * activation_derivative(y_1)
#    d_0 = (np.dot(W_h.T, d_0_unrolled[-1]) + np.dot(W_1.T, d_1)) * activation_derivative(h)
    d_0 = (np.dot(W_1.T, d_1)) * activation_derivative(h)

    d_0_unrolled.pop(0); d_0_unrolled.append(d_0)
    d_1_unrolled.pop(0); d_1_unrolled.append(d_1)
    h_unrolled.pop(0); h_unrolled.append(h)
    inputs_unrolled.pop(0); inputs_unrolled.append(sample)

    for l in range(0, backprop_depth):
        d_1_l += np.dot(d_1_unrolled[l], h_unrolled[l].T)
        d_h_l += np.dot(d_0_unrolled[l], h_unrolled[l].T)
        d_0_l += np.dot(d_0_unrolled[l], inputs_unrolled[l].T)

    W_0 += nu * d_0_l
    W_1 += nu * d_1_l
    W_h += nu * d_h_l
    d_0_l *= 0
    d_1_l *= 0
    d_h_l *= 0
    if(j % 10000 == 0):
        print('Processed samples ' , j, 'of ', train_data_len)

actual_out = np.zeros((valid_data_len, 1), np.float32)
h_prev = np.zeros((hidden_dim, 1), np.float32)
for j in range(valid_data_len):
    sample = Xv[j, :].reshape((input_dim, 1))
    t = yv[j, 0].reshape((output_dim, 1))
    y_1, h = forvard_pass(sample, W_0, W_1, W_h, h_prev)
    #e = t - y_1 # TODO plot e, and validation data
    np.copyto(h_prev, h)
    actual_out[j, 0] = y_1[0, 0]

plt.plot(yv[:, 0])
plt.plot(actual_out[:, 0], 'r')
plt.show()

