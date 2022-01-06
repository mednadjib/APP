import sys
import os 
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
os.environ['DISPLAY'] = ':0' # Solution of "cannot connect to X server" whene excute cv2.imshow()
from tensorflow.keras import Input, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import  img_to_array
import numpy as np
import math

base_model = VGG16(weights= 'imagenet', include_top = False)

class k_rpn:
    def __init__(self, path, regions):
        self.regions = regions
        self.image   = cv2.imread(path)

    #def process(self):
    # ---------------------------------------key points extraction  -------------------------------
    def kpoints_detection(self, image):
        img1 = cv2.GaussianBlur(image,(3,3),0)
        cv2.imwrite('img_org.jpg',img1)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(100)
        kp, des = surf.detectAndCompute(img1, None)   
        print('{}', 'key points detected .....').format(len(kp))
        kp_xy = []
        
        # Extract key points coordinates
        for pt in kp:
            kp_xy.appen([pt.pt[0], pt.pt[1]])
        
        return kp_xy    

    # -------------------------------------- Anchors generation -----------------------------------

    def anchors_generation(self,  kpoints):
        
        '''aspect_rtio = w/h
           areas       = h*W '''
        areas        = [64, 256, 1024]    # for 8*8, 16*16, 32*32 anchors 
        aspect_ratio = [1/2, 1, 2]

        # find height and width of anchors
        dimention = []
        for area in areas:
            for ar in aspect_ratio:
                w = math.sqrt(area*ar)
                h = math.sqrt(area/ar)
                dimention.append([w, h]) 

        anchors = []
        
        # generate 9 anchors for each keypoint location (3 scales * 3 aspect ratios) 
        for point in kpoints:
            x_c = point[0]
            y_c = point[1]
            for dim in dimention:
                w = dim[0]
                h = dim[1]
            
                x_min = x_c - (h/2)
                y_min = y_c - (w/2)
                x_max = x_c + (h/2)
                y_max = y_c + (w/2)
                
                anchors.append([x_min, y_min, x_max, y_max])
        # discard anchors outside of the image 
        return anchors


