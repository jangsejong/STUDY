# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.grid()
plt.show()