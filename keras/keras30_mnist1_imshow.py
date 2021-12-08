import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)           #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)           #(10000, 28, 28) (10000,)


print(x_train[0])
print('y_train[0]번째 값 :', y_train[0])           #y_train[0]번째 값 : 5

import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()
#import.matplotlib.pyplot.as.plt
#plt.imshow(x_train[0], 'gray')



