from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

model = Sequential()
model.add(Dense(3, input_shape=(1,)))
model.add(Dense(2))
model.add(Dense(1))

for layer in model.layers:
    print(layer.name, layer.trainable)

# model.layers[0].trainable = False
model.layers[1].trainable = False
# model.layers[2].trainable = False

model.summary()
