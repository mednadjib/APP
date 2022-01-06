import sys
import numpy as np 
import keras
from tensorflow.keras.layers import Layer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ROIPoolingLayer(Layer):
    def __init__(self, pooled_h, pooled_w, **kwargs):
        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        super(ROIPoolingLayer, self).__init__(**kwargs)
# ------------------------------------------------------------------------------------------------    
    # compute output shape 
    def output_shape(self, input_shape):
        fm_shape, rois_shape = input_shape
        assert fm_shape[0] == rois_shape[0]
        
        b_size     = fm_shape[0]
        n_rois     = rois_shape[1]
        pooled_h   = self.pooled_h
        pooled_w   = self.pooled_w
        n_channels = fm_shape[3]
        return(b_size, n_rois, pooled_h, pooled_w, n_channels)
# ------------------------------------------------------------------------------------------------    
    def call(self, inputs):
       
        """
        --> inputs[0]: feature_map tensor, 
                            shape (batch_size, pooled_height, pooled_width, n_channels)
        
        --> inputs[1]: tensor of rois from candidate bondingboxes,
                            shape (batch_size, n_rois, 4)
                            eatch roi is defined by relative coordinates (xmin, ymin, xmax, ymax)
                            between 0, 1
        """
        def extract_pool_rois(inputs):
            
            return ROIPoolingLayer.pool_rois(inputs[0], inputs[1], self.pooled_h, self.pooled_w)
        
        pooled_areas = tf.map_fn(extract_pool_rois, inputs, dtype=tf.float32)

        return pooled_areas    
# ------------------------------------------------------------------------------------------------
    # Applies RoiPoolingLayer for a single image and a single roi
    def single_poolroi(feature_map, roi, pooled_height, pooled_width):
        
        fm_h = feature_map.shape[0]
        fm_w = feature_map.shape[1]
        
        h_start = tf.cast(fm_h * roi[0], 'int32')
        w_start = tf.cast(fm_w  * roi[1], 'int32')
        h_end   = tf.cast(fm_h * roi[2], 'int32')
        w_end   = tf.cast(fm_w  * roi[3], 'int32')
        
        region = feature_map[h_start:h_end, w_start:w_end, :]

        # Divide the region into non overlapping areas
        region_height = h_end - h_start
        region_width  = w_end - w_start
        
        h_step = tf.cast(region_height / pooled_height, 'int32')
        w_step = tf.cast(region_width  / pooled_width , 'int32')
        
        areas = [[(i*h_step, j*w_step, 
                (i+1)*h_step if i+1 < pooled_height else region_height, 
                (j+1)*w_step if j+1 < pooled_width else region_width)
                for j in range(pooled_width)] 
                for i in range(pooled_height)]
        
        # take the maximum in eatch erea and stack the result
        def pool_area(x): 
          return tf.math.reduce_max(region[x[0]:x[2], x[1]:x[3], :], axis=[0,1])
        
        pooled_features = tf.stack([[pool_area(x) for x in row] for row in areas])
        
        return pooled_features        
# ------------------------------------------------------------------------------------------------
    
    #Pooling many rois from single image
    
    def pool_rois(feature_map, rois, pooled_h, pooled_w):
        
        def extract_pool_roi(roi):
            print("roi =", roi)
            
            return ROIPoolingLayer.single_poolroi(feature_map, roi, pooled_h, pooled_w)
        
        #stack the result
        pooled_area = tf.map_fn(extract_pool_roi, rois, dtype=tf.float32)
        
        return pooled_area  
# ---------------------------------  the rest of code for testing --------------------------------
'''# Define parameters
batch_size = 1
img_height = 200
img_width = 100
n_channels = 1
n_rois = 2
pooled_height = 7
pooled_width = 7

feature_maps_shape = (batch_size, img_height, img_width, n_channels)
# placeholder: vaiable that allows us to ceate opeations, and build computiation graph without data
feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)

feature_maps_np = np.ones(feature_maps_tf.shape, dtype='float32')
feature_maps_np[0, img_height-1, img_width-3, 0] = 50

# Create batch size
roiss_shape = (batch_size, n_rois, 4)
roiss_tf    = tf.placeholder(tf.float32, shape=roiss_shape)
roiss_np    = np.asarray([[[0.5,0.2,0.7,0.4], [0.0,0.0,1.0,1.0]]], dtype='float32')
# Create layer
roi_layer = ROIPoolingLayer(pooled_height, pooled_width)

pooled_features = roi_layer([feature_maps_tf, roiss_tf])
print(f"output shape of layer call = {pooled_features.shape}")

# Run tensorflow session
with tf.Session() as session:
    result = session.run(pooled_features, 
                         feed_dict={feature_maps_tf:feature_maps_np,  
                                    roiss_tf:roiss_np})
    
print(f"result.shape = {result.shape}")
print(f"first  roi embedding=\n{result[0,0,:,:,0]}")
print(f"second roi embedding=\n{result[0,1,:,:,0]}")'''



        


       
