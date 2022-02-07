import numpy as np



f = lambda x: x**2 -4*x +6

def gradient2(x):
    return  2*x -4

def gradient3(x):
    temp = x**2 -4*x +6
    return temp


x = 10
epochs = 20
learning_rate = 0.2



print("step\t x\t f(x)")
print("{:02d}" .format(0), "\t", "{:.2f}" .format(x))
print("{:02d}\t {:6.5f}\t {:6.5f}\t ".format(0, x, f(x) ))

for i in range(epochs):
    x = x- learning_rate * gradient2(x)
    
    print("{:02d}\t {:6.5f}\t {:6.5f}\t ".format(i+1, x, f(x) ))
    
    
