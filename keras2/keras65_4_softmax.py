# 난 정말 시그모이드!

from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(1, 5)
y = softmax(x)

ratio = y
labels = y

plt.pie(y, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)  #autopct 는 백분율 표시를 위한 형식
plt.show()
