# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    # return np.maximum(0, x)
    return np.where(x > 0, x, 0)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()