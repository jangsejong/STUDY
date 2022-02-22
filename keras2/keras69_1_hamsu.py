from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, GlobalAveragePooling2D
from tensorflow.keras.application import VGG16, VGG19, Xception


input = Input(shape=(100,100,3))
VGG16_model = VGG16(include_top=False, input_shape=(100,100,3))
