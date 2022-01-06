import keras, os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.utils.vis_utils import plot_model

import numpy as np
import json
base_model = VGG16(weights= 'imagenet', include_top = True) 
plot_model(base_model, to_file='vgg.png')


