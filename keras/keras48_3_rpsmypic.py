#

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

pic_path = '../_data/image/cat_dog/KakaoTalk_20211227_203446096.jpg'
model_path = '../_save_npy/keras48_3_save_npy.h5'

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

# if __name__ == '__main__':
#     model = load_model(model_path)
#     new_img = load_my_image(pic_path)
#     pred = model.predict(new_img)
#     paper = pred[0][0]*100
#     rock = pred[0][1]*100
#     scissors = pred[0][1]*100
#     if haman > hourse:
#         print(f"당신은 {round(haman,2)} % 확률로 사람 입니다")
#     elfe:
#         print(f"당신은 {round(hourse,2)} % 확률로 말 입니다")   
#     else:
#         print(f"당신은 {round(hourse,2)} % 확률로 말 입니다")
if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    pred = model.predict(new_img)
    classes0 = pred[0][0] * 100
    classes1 = pred[0][1] * 100
    classes2 = pred[0][2] * 100
    print(classes0,classes1,classes2)
    # print(max(classes0,classes1,classes2))

    if max(classes0,classes1,classes2) == classes0:
        print(f"{round(classes0, 2)} % 확률로 보 입니다")
    elif max(classes0,classes1,classes2) == classes1:
        print(f"{round(classes1, 2)} % 확률로 바위 입니다")
    else:
        print(f"{round(classes2, 2)} % 확률로 가위 입니다")


'''
100.0 % 확률로 바위 입니다

'''

