import os
import gc
import sys

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
from skimage import color

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
import keras.backend as K
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D 
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50

from PIL import Image
from tqdm import tqdm
import random as rnd
import cv2
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

# !pip install livelossplot
from livelossplot import PlotLossesKeras

# %matplotlib inline

train_path = '../input/railway-track-fault-detection/Railway Track fault Detection Updated/Train'
val_path = '../input/railway-track-fault-detection/Railway Track fault Detection Updated/Validation'
test_path = '../input/railway-track-fault-detection/Railway Track fault Detection Updated/Test'

train_df_defective = pd.DataFrame(os.listdir(train_path+'/Defective'))
val_df_defective = pd.DataFrame(os.listdir(val_path+'/Defective')) 
test_df_defective = pd.DataFrame(os.listdir(test_path+'/Defective')) 
train_df_undefective = pd.DataFrame(os.listdir(train_path+'/Non defective'))
val_df_undefective = pd.DataFrame(os.listdir(val_path+'/Non defective')) 
test_df_undefective = pd.DataFrame(os.listdir(test_path+'/Non defective')) 
defective=pd.concat([train_df_defective,test_df_defective,val_df_defective], axis=0)
undefective=pd.concat([train_df_undefective,test_df_undefective,val_df_undefective], axis=0)
print('Train samples defective: ', len(train_df_defective))
print('Val samples defective: ', len(val_df_defective))
print('Test samples defective: ', len(test_df_defective))
print()
print('Train samples undefective: ', len(train_df_undefective))
print('Val samples undefective: ', len(val_df_undefective))
print('Test samples undefective: ', len(test_df_undefective))


widths, heights = [], []
defective_path_images = []

for path in tqdm(defective[0]):
    try:
        width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Test/Defective/'+path).size
        widths.append(width)
        heights.append(height)
        defective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Test/Defective/'+path)
    except:
        try:
            width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Train/Defective/'+path).size
            widths.append(width)
            heights.append(height)
            defective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Train/Defective/'+path)
        except:
            try:
                width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Validation/Defective/'+path).size
                widths.append(width)
                heights.append(height)
                defective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Validation/Defective/'+path)
            except:
                continue
    
df_defective = pd.DataFrame()
df_defective["width"] = widths
df_defective["height"] = heights
df_defective["path"] = defective_path_images
df_defective["dimension"] = df_defective["width"] * df_defective["height"]


df_defective.sort_values('width').head(84)

df_defective.head(84).mean()

widths, heights = [], []
undefective_path_images = []

for path in tqdm(undefective[0]):
    try:
        width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Test/Non defective/'+path).size
        widths.append(width)
        heights.append(height)
        undefective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Test/Non defective/'+path)
    except:
        try:
            width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Train/Non defective/'+path).size
            widths.append(width)
            heights.append(height)
            undefective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Train/Non defective/'+path)
        except:
            try:
                width, height = Image.open('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Validation/Non defective/'+path).size
                widths.append(width)
                heights.append(height)
                undefective_path_images.append('../input/railway-track-fault-detection/Railway Track fault Detection Updated/Validation/Non defective/'+path)
            except:
                continue
    
df_undefective = pd.DataFrame()
df_undefective["width"] = widths
df_undefective["height"] = heights
df_undefective["path"] = undefective_path_images
df_undefective["dimension"] = df_undefective["width"] * df_undefective["height"]

df_undefective["dimension"]

df_undefective.sort_values('width').head(84)

df_undefective.head(84).mean()

def is_grey_scale(givenImage):
    w,h = givenImage.size
    for i in range(w):
        for j in range(h):
            r,g,b = givenImage.getpixel((i,j))
            if r != g != b: return False
    return True

sampleFrac = 0.4
#get our sampled images
isGreyList = []
for imageName in df_undefective["path"].sample(frac=sampleFrac):
    val = Image.open(imageName).convert('RGB')
    isGreyList.append(is_grey_scale(val))
print(np.sum(isGreyList) / len(isGreyList))
del isGreyList

sampleFrac = 0.4
#get our sampled images
isGreyList = []
for imageName in df_defective["path"].sample(frac=sampleFrac):
    val = Image.open(imageName).convert('RGB')
    isGreyList.append(is_grey_scale(val))
print(np.sum(isGreyList) / len(isGreyList))
del isGreyList

def get_rgb_men(row):
    img = cv2.imread(row['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.sum(img[:,:,0]), np.sum(img[:,:,1]), np.sum(img[:,:,2])

tqdm.pandas()
df_defective['R'], df_defective['G'], df_defective['B'] = zip(*df_defective.progress_apply(lambda row: get_rgb_men(row), axis=1) )

def get_rgb_men(row):
    img = cv2.imread(row['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.sum(img[:,:,0]), np.sum(img[:,:,1]), np.sum(img[:,:,2])

tqdm.pandas()
df_undefective['R'], df_undefective['G'], df_undefective['B'] = zip(*df_undefective.progress_apply(lambda row: get_rgb_men(row), axis=1) )

def show_color_dist(df, count):
    fig, axr = plt.subplots(count,2,figsize=(15,15))
    if df.empty:
        print("Image internsity of selected color is weak")
        return
    for idx, i in enumerate(np.random.choice(df['path'], count)):
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axr[idx,0].imshow(img)
        axr[idx,0].axis('off')
        axr[idx,1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2]))) 
        x, y = np.histogram(img[:,:,0], bins=255)
        axr[idx,1].bar(y[:-1], x, label='R', alpha=0.8, color='red')
        x, y = np.histogram(img[:,:,1], bins=255)
        axr[idx,1].bar(y[:-1], x, label='G', alpha=0.8, color='green')
        x, y = np.histogram(img[:,:,2], bins=255)
        axr[idx,1].bar(y[:-1], x, label='B', alpha=0.8, color='blue')
        axr[idx,1].legend()
        axr[idx,1].axis('off')

df = df_defective[((df_defective['B']*1.05) < df_defective['R']) & ((df_defective['G']*1.05) < df_defective['R'])]
show_color_dist(df, 8)

df = df_undefective[((df_undefective['B']*1.05) < df_undefective['R']) & ((df_undefective['G']*1.05) < df_undefective['R'])]
show_color_dist(df, 8)

df = df_defective[(df_defective['G'] > df_defective['R']) & (df_defective['G'] > df_defective['B'])]
show_color_dist(df, 8)

df = df_undefective[(df_undefective['G'] > df_undefective['R']) & (df_undefective['G'] > df_undefective['B'])]
show_color_dist(df, 8)

df = df_defective[(df_defective['B'] > df_defective['R']) & (df_defective['B'] > df_defective['G'])]
show_color_dist(df, 8)

df = df_undefective[(df_undefective['B'] > df_undefective['R']) & (df_undefective['B'] > df_undefective['G'])]
show_color_dist(df, 8)

gc.collect()