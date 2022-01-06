from re import I
import sys
import numpy as np 
import keras
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.backend import map_fn
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class IoULayer(Layer):
    def __init__(self, **kwargs):
        super(IoULayer, self).__init__(**kwargs)
        
        
# -------------------------------------------------------------------------------------------------
    
    def call(self, inputs):
        '''
        inputs contrain two tensors of all valid combination bitween ground truth boxes and kpoint_boxes boxes
        ''' 
           
        def multiple_IoU(inputs):
            
            
            iou_anchor = IoULayer.get_single_IoU(inputs[0], inputs[1])
            
            return inputs
        
        
        
        IoU_anchors = tf.map_fn(multiple_IoU, inputs, fn_output_signature=tf.float32)
        #IoU_anchors = tf.map_fn(multiple_IoU, inputs, fn_output_signature=tf.int32)
        
        return IoU_anchors
    
    
    def get_single_IoU(lst_bbox1, lst_bbox2):
        iou_list = []
        
        print(len(lst_bbox1))
        for gt_bbx, anchor in zip(lst_bbox1,lst_bbox2):
            x_left   = tf.math.maximum(gt_bbx[0], anchor[0])
            y_top    = tf.math.maximum(gt_bbx[1], anchor[1])
            x_right  = tf.math.minimum(gt_bbx[2], anchor[2])
            y_bottom = tf.math.minimum(gt_bbx[3], anchor[3])

            bb1_area = (gt_bbx[2]- gt_bbx[0])*(gt_bbx[3]- gt_bbx[1])
            anchor_area = (anchor[2]- anchor[0])*(anchor[3]- anchor[1])

            intersect_area = tf.math.abs(tf.math.maximum((x_right - x_left), 0) * tf.math.maximum((y_bottom - y_top),0))
            iou = intersect_area / float(bb1_area + anchor_area - intersect_area)
            iou_list.append(iou)
           
        '''
        def single_iou(gt_bbx, anchor):
            

            x_left   = tf.math.maximum(gt_bbx[0], anchor[0])
            y_top    = tf.math.maximum(gt_bbx[1], anchor[1])
            x_right  = tf.math.minimum(gt_bbx[2], anchor[2])
            y_bottom = tf.math.minimum(gt_bbx[3], anchor[3])

            bb1_area = (gt_bbx[2]- gt_bbx[0])*(gt_bbx[3]- gt_bbx[1])
            anchor_area = (anchor[2]- anchor[0])*(anchor[3]- anchor[1])

            intersect_area = tf.math.abs(tf.math.maximum((x_right - x_left), 0) * tf.math.maximum((y_bottom - y_top),0))
            iou = intersect_area / float(bb1_area + anchor_area - intersect_area)
            

            return iou
           
       
        iou_list = tf.stack([single_iou(bbx1, bbx2) for bbx1, bbx2 in zip(lst_bbox1, lst_bbox2)] )'''
        return iou_list
       

