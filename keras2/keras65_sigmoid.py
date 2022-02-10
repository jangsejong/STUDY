# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5, 5, 0.1)
print(len(x))

y = sigmoide(x)

plt.plot(x, y)
plt.grid()
plt.show()