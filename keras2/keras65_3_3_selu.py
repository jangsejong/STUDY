# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def selu(x):
    return (x>0)*x + (x<=0)*(0.01*x)
    # return (np.exp(x) - 1) * 1.6732632423543772848170429916717
    # return np.where(x > 0, x, 1.0507 * np.exp(x) - 1.0507)

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()