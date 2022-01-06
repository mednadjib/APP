import numpy as np 
import random
import sys
#from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 
import os
'''
#os.environ['DISPLAY'] = ':0'''
'''img = np.zeros(600).reshape(30, 20)

for i in range(30):
    for j in range(20):
        img[i, j] = random.randint(0, 256)

def inputfunct(x):
    return 0.25*(np.sin(2*np.pi*x*x)+2.0)

np.random.seed(5)
X = np.random.sample([2048])

Y = inputfunct(X) + 0.2*np.random.normal(0,0.2,len(X))

Xreal = np.arange(0.0, 1.0, 0.01)
Yreal = inputfunct(Xreal)
### Model creation: adding layers and compilation
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
nepoch =500
nbatch = 16
model.fit(X, Y, epochs=nepoch, batch_size=nbatch)
Ylearn = model.predict(Xreal)

### Make a nice graphic!
plt.plot(X,Y,'.', label='Raw noisy input data')
plt.plot(Xreal,Yreal, label='Actual function, not noisy', linewidth=4.0, c='black')
plt.plot(Xreal, Ylearn, label='Output of the Neural Net', linewidth=4.0, c='red')
plt.legend()
plt.savefig('neural-
-keras-function-interpolation.png')
plt.show()'''
'''import tensorflow as tf
from tensorflow import keras
strategy = tf.distribute.MirroredStrategy()'''
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
'''bs = 5

tag_lens    = np.zeros((bs, ), dtype=np.int32)
tl_tags     = np.zeros((bs, 15))
detect = [ 1, 2,3,4,5,6,7,8]
for b_ind in range(bs): 
    print('b_ind =', b_ind)
    print('The {} image'.format(b_ind))
    for ind in range(len(detect)):
        xtl = random.randint(1, 10)
        tag_ind = tag_lens[b_ind]
        print(tag_ind)
        tl_tags[b_ind,tag_ind] = xtl
        tag_lens[b_ind] += 1
        print('tag_lens =', tag_lens)
    print(tl_tags)    
    print("-----------------------------")'''
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

'''np.random.seed(1)
n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = im > im.mean()

label_im, nb_labels = ndimage.label(mask)

print('Find the largest connect component')
sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
mask_size = sizes < 1000
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = 0
labels = np.unique(label_im)
label_im = np.searchsorted(labels, label_im)

# Now that we have only one connect component, extract it's bounding box
slice_x, slice_y = ndimage.find_objects(label_im==4)[0]
roi = im[slice_x, slice_y]

plt.figure(figsize=(4, 2))
plt.axes([0, 0, 1, 1])
plt.savefig('bbox of larger object.png')
plt.imshow(roi)
plt.axis('off')

plt.show()'''
# GRADED FUNCTION: identity_block
# ----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
# ------------------------ ResNet Identity block example ---------------------------------
# ----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
'''import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
#%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X]) 
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
tf.reset_default_graph()
#from tensorflow.python.framework import ops
#ops.reset_default_graph()

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

# GRADED FUNCTION: ResNet50

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

#Run the following code to build the model's graph. If your implementation is not 
#correct you will know it by checking your accuracy when running model.fit(...) below.

model = ResNet50(input_shape = (64, 64, 3), classes = 6)

#As seen in the Keras Tutorial Notebook, prior training a model, you need to configure 
#the learning process by compiling the model.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# loding dataset...
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
à
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))'''
"""from tensorflow.keras import Input, Model
ff = Input(shape = (3, 2, 4, 2))
for i in range(5):
    l = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11])
    s = []
    data_rng = None
    if data_rng == None:
        data_rng = np.random.RandomState(os.getpid())
    
    print('shuffling index......')
    rand_perm = data_rng.permutation(len(l))  
    print('rand_perm = ', rand_perm)
    l =l[rand_perm]    
    print(l)

import cv2
print(cv2.__version__)   """
'''a = [[1, 2, 3,10,12], [4,5, 6,15,9], [7,8,9,80,21],[10,11,12,1,0]]   
reg = np.squeeze(a)[1:4, 2:4]
aa = [3, 4, 6, 12]    
xmin, ymin, xmax, ymax = aa
print(xmin)
print(ymin)
print(xmax)
print(ymax)
import tensorflow.compat.v1 as tf 
"""files = os.listdir(paths)
for file in files:
    path_file = os.path.join(paths,)
"""
ss = []
for i in range( 100):
    s =  i*i 
    ss.append(s)  
tf.stack
           '''
import os
import json     
paths = '/home/imene/APP-M/data/coco/cache/MSCOCO_anchors1/anch_7.json'
with open(paths, 'r') as f:
    data = json.loads(f.read())
    
for k, v in data.items():
    print(k)
    print('-------------------------------------------------------------------------')
    print(len(v[0]))
    print(len(v[1]))