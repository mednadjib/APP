import sys
import numpy as np 
import keras
from tensorflow.keras.layers import Layer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class ROIPoolingLayer(Layer):
    def __init__(self , pooled_h, pooled_w,  **kwargs ):

        """
        tf: (height, weidth, channels)
        th: (channels, height, width)
        """
        
        self.pool_size = (pooled_h, pooled_w)
   # --------------------------------------------------------------------------------------------- 
    def call (self, inputs):
        """
        inputs[0]: feature map tensor
        inputs[1]: rois batch of shape (batch_size, n_rois, 4 )
                  roi is identified by relative coordinates (xmin, ymin, xmax, ymax)  
                  between (0, 1)
        """ 
        def pooled_rois (inputs):

            return ROIPoolingLayer.get_regions(inputs[0], inputs[1], self.pool_size)
        
        pooled_areas =  tf.map_fn(pooled_rois, inputs, dtype = tf.float32)   
    # --------------------------------------------------------------------------------------------
    def get_regions(self, feature_map, rois, pool_size):
        
        def extract_region(roi):
            return ROIPoolingLayer.get_single_region(feature_map, roi, pool_size)  
                 #stack the result
        pooled_area = tf.map_fn(extract_region, rois, dtype=tf.float32)

    def get_single_region(feature_map, roi, pool_size ):

        fm_h = feature_map.shape[0]
        fm_w = feature_map.shape[1]


        pool_h = pool_size[0]
        pool_w = pool_size[1]

        
        h_start = tf.cast(fm_h * roi[0], 'int32')
        w_start = tf.cast(fm_w  * roi[1], 'int32')
        h_end   = tf.cast(fm_h * roi[2], 'int32')
        w_end   = tf.cast(fm_w  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]
        pool = np.zeros(pool_h, pool_w)

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        
        h_step = tf.cast(region_height / pool_h, 'int32')
        w_step = tf.cast(region_width  / pool_w, 'int32')
        
        areas = [[(i*h_step, j*w_step, 
                (i+1)*h_step if i+1 < pool_h else region_height, 
                (j+1)*w_step if j+1 < pool_w else region_width)
                for j in range(pool_w)] 
                for i in range(pool_h)]

        # take the maximum in eatch erea and stack the result
        def pool_area(x):
            for i in range(pool_w):
                for j in range(pool_h):
                    pool[i, j] = tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
            return pool        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])  
        
        return pooled_features 
                 




        
        









    




        
