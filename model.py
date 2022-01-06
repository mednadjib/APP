import sys
import os
import cv2
import random
import numpy as np 
from config import  conf
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import  tensorflow.keras.preprocessing as Tpre


mdl = VGG16(weights= 'imagenet', include_top = False)
# include_top = False:  to drop out the last 3 layers: flatten + 2 fc 


#The inputs of RoiPoolingLayer: feature_map, rois, ind of feature_maps
# feature_map shape = [batch_size, h, w, n_chanels]
fature_map = tf.keras.Input(shape=(None, None, None, 1536)) 

# rois shape = [batch_size, n_rois, 4] 4 for [xmin, ymin xmax, ymax]
rois       = tf.keras.Input(shape =( None, 4))

# Ind shape = [batch_size, ] this input indicate which feature map those rois are belong to
Ind        = tf.keras.Input(shape =( None, 1))


   def __init__(self, Dname, imsubset, detection): 
    
      size        = conf._input_size
      b_size      = conf.batch_size
      categories  = DataS.categories(Dname)
      class_id    = DataS.class_id(Dname)
   
      # Network inputs
      images      = np.zeros((b_size, 3, size[0], size[1]))
      keypoints   = np.zeros((b_size, categories, 300, 7))
      descriptors = np.zeros((b_size, categories, 300, 128))
      ltp_histo = np.zeros((b_size, categories, 300, 128))
      cont_reg = np.zeros((b_size, categories))
      #Network s
      bboxes   = np.zeros((b_size, categories, 10, 5))
      
      #random.shuffle(imsubset)
      img_dict = {ind+1: fname for ind, fname in enumerate(imsubset ) }
      
      id_img = 6
      path_img = img_dict[id_img]
      
      #VGG16 feature map 
      name_img = path_img.split('/')[-1]
      image    = cv2.imread(path_img)
      # resize the image in to appropriate format of VGG16 inputs.
      x = cv2.resize(image, (224, 224))
      
      # keras works with batches of images, the format of the image must be 
      #(simples, size1, size2, channels), not ( size1, size2, channels) so we need to add an 
      # aditionnal dimension.
      # expand_dims() is used to add the number of images: x.shape = (1, 224, 224, 3)
      
      # preprocess_input(): subtracts the mean RGB channels of the imagenet dataset.
      # This is because the model you are using has been trained on a different dataset 
      
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      fp = mdl.predict(x)
      
      # create model
      feature_map = Input(batch_shape=(None,None,None,1536))
      rois        = Input(batch_shape=(None, 4))
      ind         = Input(batch_shape=(None, 1),dtype='int32')    
      
      base_anchor = np.array([1, 1, 16, 16]) - 1
      print(base_anchor)
      sys.exit()

      w, h = image.shape[0], image.shape[1]
      regions= detection[path_img]
      train_images     = texture(name_img, image, regions)
      
      
      
      '''for id_batch in range(b_size):
         id_img   = (id_img + 1) % len(imsubset)
         path_img = img_dict[id_img]
         image    = cv2.imread(path_img)
         w, h = image.shape[0], image.shape[1]
      
         # resize image 
         #image            = cv2.resize(image, (size[0], size[1]))
         x_ratio = size[0]/w
         y_ratio = size[1]/h
         
         #images[id_batch] = image.transpose((2,0,1)) 
         
         regions= detection[path_img]
         kp     = texture.SURF(path_img, image, regions)
         #descript = texture.Motif(image, kp, regions)
         #texture.reg_image(image, regions)
         
         for descr_reg, detect in zip(descript, regions): 
            xtl = detect[0]
            ytl = detect[1]
            
            #Find new position of region coordinates
            new_xtl = round(x_ratio*xtl)
            new_ytl = round(y_ratio*ytl)
            
            category = class_id[detect[-1]]
            
            cor = cont_reg[id_batch, category]
            bboxes[id_batch,category, cor] = [new_xtl, new_ytl, detect[2], detect[3], category]
            cont_reg[id_batch, category]+=1

            i = 0
            for pt in descr_reg:
               
               keypoints[id_batch, category, i ] = [pt[0][0],pt[0][1],pt[0][2],pt[0][3],pt[0][4],pt[0][5],pt[0][6]]
                  
               descriptors[id_batch, category, i ]= pt[0][-1]
                  
               ltp_histo[id_batch, category, i]  = pt[1]
               i+=1'''
      
      
      self.input = [images, keypoints, descriptors, ltp_histo]   
      self.output = [bboxes]      
         
               
         
         

         
         
         
      
      



 
 
