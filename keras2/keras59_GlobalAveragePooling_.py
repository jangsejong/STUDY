import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import print_summary, to_categorical
from keras import backend as K
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
                     
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')        
        
BATCH_SIZE = 100
NUM_CLASSES = 100
EPOCHS = 165000
INIT_DROPOUT_RATE = 0.5
MOMENTUM_RATE = 0.9
INIT_LEARNING_RATE = 0.01
L2_DECAY_RATE = 0.0005
CROP_SIZE = 32
LOG_DIR = './logs'
MODEL_PATH = './keras_cifar100_model.h5'  

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)        
print(y_train.shape,y_test.shape)     
        
x_train = x_train.astype('float32')       
x_test = x_test.astype('float32')      
x_train /= 255.0
x_test /= 255.0


model = Sequential()
model.add(ZeroPadding2D(4, input_shape=x_train.shape[1:]))

# Stack 1:
model.add(Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

# Stack 2:
model.add(Conv2D(384, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(384, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

# Stack 3:
model.add(Conv2D(640, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

                   # Stack 4:
model.add(Conv2D(768, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

                   # Stack 5:
model.add(Conv2D(896, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

                   # Stack 6:
model.add(Conv2D(1024, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Conv2D(1152, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))

                   # Stack 7:
model.add(Conv2D(1152, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(Activation('elu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(INIT_DROPOUT_RATE))
model.add(Flatten())
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))


'''
def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=15, callbacks=[callback], verbose=0)
'''

def lr_scheduler(epoch, lr, step_decay = 0.1):
                   return float(lr * step_decay) if epoch == 35.000 else lr

def dr_scheduler(epoch, layers, rate_list = [0.0, .1, .2, .3, .4, .5, 0.0], rate_factor = 1.5):
                     if epoch == 85000:
                        for i, layer in enumerate([l for l in layers if "dropout" in np.str.lower(l.name)]):
                              layer.rate = layer.rate + rate_list[i]
                     elif epoch == 135000:
                          for i, layer in enumerate([l for l in layers if "dropout" in np.str.lower(l.name)]):
                                layer.rate = layer.rate + layer.rate * rate_factor if layer.rate <= 0.66 else 1
                                return layers

class StepLearningRateSchedulerAt(LearningRateScheduler):
                    def __init__(self, schedule, verbose = 0): 
                         super(LearningRateScheduler, self).__init__()
                         self.schedule = schedule
                         self.verbose = verbose

                    def on_epoch_begin(self, epoch, logs=None): 
                          if not hasattr(self.model.optimizer, 'lr'):
                              raise ValueError('Optimizer must have a "lr" attribute.')
        
                          lr = float(K.get_value(self.model.optimizer.lr))
                          lr = self.schedule(epoch, lr)
   
                          if not isinstance(lr, (float, np.float32, np.float64)):
                               raise ValueError('The output of the "schedule" function ' 'should be float.')
    
                          K.set_value(self.model.optimizer.lr, lr)
                          if self.verbose > 0: 
                                  print('\nEpoch %05d: LearningRateScheduler reducing learning ' 'rate to %s.' % (epoch + 1, lr))
#For Dropout Rate :
class DropoutRateScheduler(Callback):
                        def __init__(self, schedule, verbose = 0):
                             super(Callback, self).__init__()
                             self.schedule = schedule
                             self.verbose = verbose
     
                        def on_epoch_begin(self, epoch, logs=None):
                            if not hasattr(self.model, 'layers'):
                                raise ValueError('Model must have a "layers" attribute.')
        
                            layers = self.model.layers
                            layers = self.schedule(epoch, layers)
    
                            if not isinstance(layers, list):
                                raise ValueError('The output of the "schedule" function should be list.')
    
                            self.model.layers = layers
    
                            if self.verbose > 0:
                                for layer in [l for l in self.model.layers if "dropout" in np.str.lower(l.name)]:
                                    
                            print('\nEpoch %05d: Dropout rate for layer %s: %s.' % (epoch + 1, layer.name, layer.rate))
                            
                            
#Defining Random Crop :
def random_crop(img, random_crop_size):
                          height, width = img.shape[0], img.shape[1]
                          dy, dx = random_crop_size
                          x = np.random.randint(0, width - dx + 1)
                          y = np.random.randint(0, height - dy + 1)
                          return img[y:(y+dy), x:(x+dx), :]

#Defining crop generator function :
def crop_generator(batches, crop_length, num_channel = 3):
                            while True:
                                batch_x, batch_y = next(batches)
                                batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, num_channel))
                                for i in range(batch_x.shape[0]):
                                    batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
                                yield (batch_crops, batch_y)
#Using optimizer stochastic Gradient Descent

opt = SGD(lr=INIT_LEARNING_RATE, momentum=MOMENTUM_RATE)
            
lr_rate_scheduler = StepLearningRateSchedulerAt(lr_scheduler)
dropout_scheduler = DropoutRateScheduler(dr_scheduler)
tensorboard = TensorBoard(log_dir=LOG_DIR, batch_size=BATCH_SIZE)
checkpointer = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=1, save_best_only=True)
                     
                     
                     
# Data Augmentation   
datagen = ImageDataGenerator(samplewise_center=True,
                                                   zca_whitening=True,
                                                   horizontal_flip=True
                                                   )

datagen.fit(x_train)
                      
train_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
train_flow_w_crops = crop_generator(train_flow, CROP_SIZE)
valid_flow = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)


model.fit_generator(train_flow_w_crops,
                                          epochs=EPOCHS,
                                          steps_per_epoch=len(x_train) / BATCH_SIZE,
                                          callbacks=[lr_rate_scheduler, dropout_scheduler, tensorboard, checkpointer],
                                          validation_data=valid_flow,
                                          validation_steps=(len(x_train) / BATCH_SIZE))








