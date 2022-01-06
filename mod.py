from random import random
import sys
import cv2
import numpy as np 
import random

from tensorflow.python.training.tracking import base
#import tensorflow as tf 
import tensorflow.compat.v1 as tf
from keras.preprocessing.image import  img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from ROI.ROIPoolingLayer2 import ROIPoolingLayer
from ROI.IoULayer import IoULayer
from tensorflow.keras import Input, Model
from config import system_configs
from keras.layers import Flatten, Dense, BatchNormalization
from ROI.Texture import texture
from utils import box_resize
import os 
import json


base_model = VGG16(weights= 'imagenet', include_top = True)

class MODEL:
    def __init__(self, gt_boxes, data_size, dataset_name):

        self.db_size    = data_size
        self.dataset_name = dataset_name
        self.batch_size = system_configs.batch_size
        
        self.num_rois   = system_configs.n_rois
        self.n_channels = system_configs.n_channels
        self.pooled_h   = system_configs.pooled_h
        self.pooled_w   = system_configs.pooled_w

        self.num_cat    = system_configs.categories[dataset_name]
        self.gt_boxes   = gt_boxes
        
        self.k_ind     = 1
        self.step      = self.batch_size/ self.db_size
        
        self.anchors_dict = {}
        
    
         
# ------------------------------------------------------------------------------------------------
    def train(self):
        print('Training data.....')
        inputs, outputs =   self.worker() 
        feature_map     = inputs[0]
        rois            = inputs[1]
        feature_map_ind = inputs[2]
        nbr_rois        = len(rois)
        anchors_dict = {}
        print(nbr_rois)
        
        # Inputs of the model
        # one feature_map for each batch
        feature_map     = Input(batch_shape = (self.batch_size ,None, None,self.n_channels )) 
        rois            = Input(batch_shape = (self.batch_size, self.num_rois, 4 ))
        # indicate witch feature map, rois belong to 
        feature_map_ind = Input(batch_shape = (self.batch_size, self.num_rois, 1))  
    
        pooled_features = ROIPoolingLayer(self.pooled_h, self.pooled_w)(
                                            [feature_map, rois, feature_map_ind])
                                      
        flat            = Flatten()(pooled_features)
        fc1             = Dense(self.num_cat * nbr_rois, activation = "sigmoid")(flat)
        """fc1          = Dense( units=1024, activation="sigmoid",name="fc2")(flat)
        fc1             = BatchNormalization()(fc1)
        
        #Outtputs of the model: 
        # 1- bounding box coordinates 
        output_prob   = Dense(unit = self.num_cat * nbr_rois, activation = "softmax")(fc1)
        output_deltas = Dense(units=4 * 200, activation="linear",
                                kernel_initializer="uniform",name="deltas2")(fc1)

        # 2- categories 
        output_scores = Dense(units=1 * 200,activation="softmax",
                              kernel_initializer="uniform",name="scores2")(fc1)"""


         
        feature_map     = inputs[0]
        rois            = inputs[1]
        feature_map_ind = inputs[2]


        model = Model(inputs =[feature_map, rois,feature_map_ind], 
                      outputs = [output_deltas, output_scores])

        model.summary()
        model.compile(optimizer='rmsprop',
            loss={'deltas2':smoothL1, 'scores2':'categorical_crossentropy'})
        
        model.fit_generator(self.worker(), steps_per_epoch=int(self.step), epochs=100)
# ------------------------------------------------------------------------------------------------
    def worker(self):
        
        
        dict_img = dict({ind+1 : path for ind, path in enumerate(self.gt_boxes.keys())})
        inds      = [k for k in dict_img.keys()]

        
        for indx in range(self.batch_size):
            print(self.k_ind)
            #path  = dict_img[self.k_ind]
            path  = dict_img[120]
            
            rois  = self.gt_boxes[path]
            
            self.k_ind = (self.k_ind + 1) % self.db_size
               
            batch_rois = np.zeros((self.batch_size, len(rois), 4))
            batch_cat  = np.zeros((self.batch_size, len(rois), self.num_cat, 1))
            batch_indf = np.zeros((self.batch_size, len(rois), 1))

            # read the image and extract features
            name_img = path.split('/')[-1]
            image        = cv2.imread(path)   
    
            h_img, w_img = image.shape[0], image.shape[1]
            x = cv2.resize(image, (224, 224))                      # to be fed in the CNN
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)                          # add batch_size dimension  
           
            base_feature_map = base_model.predict(x)               # extract features
            base_model.summary()
            
            model_extractfeatures = Model(base_model.input, base_model.get_layer('fc2').output)
            fc2_features = model_extractfeatures.predict(x)
            fc2_features = fc2_features.reshape((25088,1))
            print(fc2_features.shape)
            
            cv2.imwrite('fm.jpg', fc2_features)
            xx = cv2.imread('fm.jpg')
            cv2.imshow('fm', xx)
            cv2.waitKey(1000)
            sys.exit()
            h    =  base_feature_map.shape[1]
            w    =  base_feature_map.shape[2]
            
            chan =  base_feature_map.shape[3]
            
            print('----------------------------------------------------------------')
         
            anchors_bias = texture(path, rois).process() 
            self.anchors_dict[path] = anchors_bias
            inputs = []
            gt_bbx = [rr[:4] for rr in rois]
            print(rois)
            
            '''bbx1 = tf.placeholder(tf.float32, shape=bbx1_shape)
            bbx2 = tf.placeholder(tf.float32, shape=bbx2_shape)
            anchor_iou = ioulayer([bbx1, bbx2])
            with tf.Session() as sess:
                result = sess.run(anchor_iou, feed_dict = {bbx1:bbox1, bbx2:bbox2})
            print(result)  '''  
            sys.exit()

            #######################################################
            for i in range(len(anchors)):
                batch_rois[indx ,i] = anchors[i]
                batch_cat[indx, i, categories[i]] = 1
                batch_indf[indx, i] = indx


            # network inputs
            f     = np.zeros((len(rois), h, w, chan))
            f[0]  = base_feature_map[0]
            r     = np.asarray(batch_rois)
            f_ind = np.asarray(batch_indf)

            # network outputs 
            gt_b, _ = box_resize(rois, h_img, w_img)
            gt_b    = np.asarray(gt_b)
            categ   = np.asarray(batch_cat)
            inputs  = [f, r, f_ind]
            outputs = [gt_b, categ]
            
            yield inputs , outputs

    def save_anchors(self, path, anchors):

        anchor_file = os.path.join('./APP-M/', '{}_anchors.json').format(self.dataset_name)

        if not os.path.exists(anchor_file):
            os.mkdir(anchor_file) 
            
        
        with open(anchor_file, 'w') as f:
            json.dump([self.images, self.detections], f, indent = 4)




            


            
            
            
            

        







   