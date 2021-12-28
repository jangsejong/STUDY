#

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/cat_dog/KakaoTalk_20211227_203446096.jpg'
model_path = '../_save_npy/keras48_1_save_weights.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(15,15))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    cat = pred[0][0]*100
    dog = pred[0][1]*100
    if cat > dog:
        print(f"당신은 {round(cat,2)} % 확률로 고양이 입니다")
    else:
        print(f"당신은 {round(dog,2)} % 확률로 개 입니다")


'''
acc :  1.0
val_acc :  1.0
loss :  1.085403875068433e-13
val_loss :  6.1267660279687414e-27

'''

