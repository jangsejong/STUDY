# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def Leaky_Relu(x):
    return (x>0)*x + (x<=0)*(0.01*x)
    # return np.maximum(0.01 * x, x)
    # return np.where(x > 0, x, 0.01 * x)

x = np.arange(-5, 5, 0.1)
y = Leaky_Relu(x)

plt.plot(x, y)
plt.grid()
plt.show()