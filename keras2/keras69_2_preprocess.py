from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

model = ResNet50(weights='imagenet')

img_path = '../_data/image/cat_dog/cat_dog4.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)

print("x.shape:", x.shape) # (261, 193, 3)

import numpy as np
x = np.expand_dims(x, axis=0) 
print("x.shape:", x.shape,x) # x.shape: (1, 261, 193, 3)

x = preprocess_input(x) # preprocess_input 함수를 이용해서 이미지를 정규화한다.
print("x.shape:", x.shape) # x.shape: (1, 261, 193, 3)
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# scaler = MaxAbsScaler()

x = x.reshape(1,224,224,3)

preds = model.predict(x)
print("preds:", preds) # preds: [[0.9998]

print('결과는 :', decode_predictions(preds, top=5)[0])