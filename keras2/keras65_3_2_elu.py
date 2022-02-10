# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)
    # return (np.exp(x) - 1) * (x > 0) + x * (x <= 0)
    

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()