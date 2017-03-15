#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def gen_deltas(len_data):
    deltas = np.zeros((1, len_data), np.float32)
    i = 0 
    while i < len_data:
        a = np.random.randint(2)
        deltas[0, i] = a
        if a == 1:
            i += 11
        i += 1
    return deltas 

def gen_steps(deltas):
    steps = np.zeros_like(deltas, np.float32)
    for i in range(0, deltas.shape[1]):
        if deltas[0, i] == 1:
            steps[0, i] = np.random.rand()
    return steps

def gen_pulses(deltas, steps):
    pulses = np.zeros_like(deltas, np.float32)
    for i in range(0, deltas.shape[1]):
        if deltas[0, i] == 1:
            pulse_len = int(steps[0, i] * 10)
            if i + pulse_len < deltas.shape[1]:
                pulses[0, i:i + pulse_len] = 1
            else:
                pulses[0, i:-1] = 1
    return pulses

def gen_data(data_len):
    x = np.zeros((data_len, 2), np.float32)
    deltas = gen_deltas(data_len)
    x[:, 0] = deltas[0, :].T
    steps = gen_steps(deltas)
    x[:, 1] = steps[0, :].T
    pulses = gen_pulses(deltas, steps)
    y = np.zeros((data_len, 1), np.float32)
    y[:, 0] = pulses[0, :].T
    return x, y

if __name__ == '__main__':
    data_len = 200
    x, y = gen_data(data_len)
    plt.subplot(311)
    plt.plot(x[:, 0])
    plt.subplot(312)
    plt.plot(x[:, 1])
    plt.subplot(313)
    plt.plot(y[:, 0])
    plt.show()

