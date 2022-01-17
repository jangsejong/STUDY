#

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/cat_dog/KakaoTalk_20211227_203446096.jpg'
model_path = '../_save_npy/keras48_4_save_npy.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(150,150))
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
    men = pred[0][0]*100
    women = pred[0][1]*100
    if men > women:
        print(f"당신은 {round(men,2)} % 확률로 남자 입니다")
    else:
        print(f"당신은 {round(women,2)} % 확률로 여자 입니다")


'''
당신은 68.57 % 확률로 남자 입니다

'''

