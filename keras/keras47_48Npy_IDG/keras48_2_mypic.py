#

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/cat_dog/KakaoTalk_20211227_203446096.jpg'
model_path = '../_save_npy/keras48_2_save_weights.h5'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(300,300))
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
    haman = pred[0][0]*100
    hourse = pred[0][1]*100
    if haman > hourse:
        print(f"당신은 {round(haman,2)} % 확률로 사람 입니다")
    else:
        print(f"당신은 {round(hourse,2)} % 확률로 말 입니다")


'''
acc :  0.9988876581192017
loss :  0.010299109853804111

당신은 100.0 % 확률로 사람 입니다

'''

