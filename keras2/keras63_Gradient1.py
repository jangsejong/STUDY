import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 -4*x +6

def gradient2(x):
    return  2*x -4

def gradient3(x):
    temp = x**2 -4*x +6
    return temp


x = np.linspace(-1,6,100)
print(x, len(x))

y = f(x)

plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk', c='red')
plt.grid()
plt.ylabel('y')
plt.xlabel('x')
plt.show()