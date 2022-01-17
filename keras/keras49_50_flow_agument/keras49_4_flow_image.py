from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5, 
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size = augment_size)  #randint
# print(x_train.shape[0])                   # 60000
# print(randidx)                            # [28809 11925 51827 ... 34693  6672 48569]
# print(np.min(randidx), np.max(randidx))   # 0 59998

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)                   # (40000, 28, 28)
print(y_augmented.shape)                   # (40000, )  ?

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)


x_augmented = train_datagen.flow(x_augmented, y_augmented, #np.zeors(augment_size),
                                 batch_size=augment_size, shuffle=False
                                 ).next()[0]
print(x_augmented)
print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train)         

# print(x_train)



#x_augmented and x_train 10pic
#subplot(2,10,?)  사용
'''
x_augmented = os.listdir(x_augmented)
x_train = os.listdir(x_train)
'''


#이미지확인하기
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg



# nrows = 4
# ncols = 4

# pic_index = 0

# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8
# x_augmented_pic = x_augmented[0]
# x_train_pic = x_train[0]

# for i, img_path in enumerate(x_augmented_pic+x_train_pic):
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off')

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# plt.show()


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow((x_augmented[0]+x_train[0]), cmap='gray')
 
plt.show()
