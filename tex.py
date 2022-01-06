import os
import sys
import math
import cv2
import random
from tqdm import tqdm
import numpy as np 
from utils import box_resize
from shapely.geometry import Point
from shapely.geometry import Polygon
from dataset.mscoco import coco
from skimage import color
from skimage.segmentation import find_boundaries
os.environ['DISPLAY'] = ':0'
class texture:
    def __init__(self):
        self.imgg = "imgg"
#-----------------------------------------------------------------------------------------

    def SURF(self, path_img, img, regions):
        print('surf execution ...')
        self.name_img = path_img.split('/')[-1]
        if regions == [0,0,0,0,0]:
            selected_pt= []
        
        else:   
            print('nbr objects:', len(regions)) 
            #im1 = cv2.imread(img)
            img1 = cv2.GaussianBlur(img,(3,3),0)
            
            surf = cv2.xfeatures2d.SURF_create()
            surf.setExtended(True) # change the size of descriptor from 64 to 128
            surf.setHessianThreshold(1000)
            kp, des = surf.detectAndCompute(img1, None)
            
            #Prepare surf keypoints  to selection
            kpts = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]
                            for k in kp])
                           
            kpd = []
            
            for point, d in zip(kpts, des):
                kpd.append([point[0], point[1], point[2], point[3],
                            point[4], point[5],point[6], d])

            #select keypoints with height response
            selected_pt = self.selection(img1, regions, kpd)

            im = cv2.imread('bbox.jpg')
            cv2.drawKeypoints(im, kp, im, color=(255,255,0))
            cv2.imwrite('surf.jpg', im)
            im = cv2.imread('surf.jpg')
            '''cv2.imshow('surf',im )
            cv2.waitKey(1000)'''
            print('nbr of selected points: ', len(selected_pt))
            '''for pt in selected_pt:
                histos = []
                label_reg = []
                
                point = (pt[0], pt[1])
                des = pt[-1]
                reg, bbox_reg = self.region(img, point)
                # transform to gray scal
                gray = color.rgb2gray(reg)
                hist = self.relaxed_ltp8_1(gray)
                similar = []
                
                for pt1 in tqdm(selected_pt):
                    if pt != pt1:
                        point1 = (pt1[0], pt1[1])
                        des1 = pt1[-1]
                        reg1, bbox_reg1 = self.region(img, point1)
                        gray1 = color.rgb2gray(reg1)
                        hist1 = self.relaxed_ltp8_1(gray1)
                        hist_sim = sum([min(a, b) for a, b in zip(hist, hist1)])
                        des_sim = sum([min(a, b) for a, b in zip(des, des1)])
                        similar.append([point1, hist_sim, des_sim])
           
                          
                hsimilar = sorted(similar,  key=lambda x: x[1], reverse=True) 
                dsimilar = sorted(similar,  key=lambda x: x[2], reverse=True)'''
            self.anchors_generation(selected_pt, regions, img) 
# ------------------------------------------------------------------------------------------------                   
    
    def selection(self,image,  regions, kpd):
        selected_pt =[]
        cont =0
        
        for reg in regions:
            
            gt_bbox = self.gt_bbox(reg)
            xmin = gt_bbox[0]
            ymin = gt_bbox[1]
            xmax = gt_bbox[2]
            ymax = gt_bbox[3]
            imgg=cv2.rectangle(image,(xmin,ymin),(xmax,ymax), color =(224, 230, 45), thickness= 5)
            cv2.imwrite('bbox.jpg', imgg)
            sort_kpd = sorted(kpd,  key=lambda x: x[4], reverse=True)  # sort points based on espence value
        # select 200 first keypoints 
        #selected_pt = [x for index, x in enumerate(sort_kpd) if index < N]
        selected_pt = sort_kpd
        return selected_pt    
    
# ----------------------------------- get ground truth bbox --------------------------------------
    def gt_bbox(self, region):
            w = region[2]
            h = region[3]
            xmin = int(region[0])
            ymin = int(region[1])
            # test if pascal_voc dataset
            if 'COCO' not in self.name_img: 
                xmax = int(w)
                ymax = int(h)
            else:    
                xmax = xmin + int(w)
                ymax = ymin + int(h)
            gt_bbox = [xmin, ymin,  xmax, ymax]    

            return gt_bbox

# -------------------- calculate intersection over union between tow regions --------------------
    def get_IoU(self, bb1, bb2):
        x_left   = max(bb1[0], bb2[0])
        y_top    = max(bb1[1], bb2[1])
        x_right  = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        bb1_area = (bb1[2]- bb1[0])*(bb1[3]- bb1[1])
        bb2_area = (bb2[2]- bb2[0])*(bb2[3]- bb2[1])
        
        intersect_area = abs(max((x_right - x_left), 0) * max((y_bottom - y_top),0))
        iou = intersect_area / float(bb1_area + bb2_area - intersect_area)
        
        return iou
# -----------------------------------------------------------------------------------------------
    def anchors_generation(self, keypoints, regions, img):
        h_obj = np.zeros((len(regions)))
        iou_obj = np.zeros((len(regions)))
        c =0
        colors = [(random.randint(0, 255),random.randint(0, 255), random.randint(0, 255) ) for _ in range(len(regions))]
        
        for reg,cl in zip(regions, colors): 
            h_cont =0
            io_cont =0
            gt_bbox = self.gt_bbox(reg)  
            xmingt = gt_bbox[0]
            ymingt = gt_bbox[1]
            xmaxgt = gt_bbox[2]
            ymaxgt = gt_bbox[3]
            rgt = img[ymingt: ymaxgt, xmingt: xmaxgt]
            # transform to grayscal 
            gray_gt = color.rgb2gray(rgt)
            # claculate relaxed_ltp histogram of ground truth region
            hgt = self.relaxed_ltp8_1(gray_gt)
            cont = 0  
            similar = []
            for pt in tqdm(keypoints):
                
                x = pt[0]
                y = pt[1]
                for pt1 in tqdm(keypoints):
                    if pt != pt1:
                        
                        xp = pt1[0]
                        yp = pt1[1]
                        xmin =int( min(x, xp))
                        ymin = int(min(y, yp))
                        xmax = int(max(x, xp))
                        ymax = int(max(y, yp))
                        bbox = [xmin, ymin, xmax, ymax] 
                        rb = img[ymin:ymax, xmin: xmax]
                        
                        gray_rb = color.rgb2gray(rb)
                        # claculate relaxed_ltp histogram of ground truth region
                        h1 = self.relaxed_ltp8_1(gray_rb)
                        
                        # similarity calculated using normalized histogram intersection 
                        sim = self.NI_similarity(hgt, h1)
                        iou = self.get_IoU(gt_bbox, bbox) 
                        '''print(sim, '------', iou)'''
                        if  (sim >= 0.5):
                            if h_cont <=10:
                                h_cont+=1
                                img = cv2.imread('bbox.jpg')
                                imgg=cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color =(255,0,255), thickness= 5)
                                h_obj[c]+=1
                                cv2.imwrite('bbox.jpg', imgg)
                                cv2.imshow('rect', img)
                                cv2.waitKey(1000)
                            else:
                                break    
                        if (iou>= 0.1):
                            if io_cont <= 10:
                                io_cont+=1
                                img = cv2.imread('bbox.jpg')
                                imgg=cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color =(0,255,0), thickness= 5)
                                iou_obj[c]+=1
                                cv2.imwrite('bbox.jpg', imgg)
                                cv2.imshow('rect', img)
                                cv2.waitKey(1000)

                            else:
                                break
                        #similar.append([sim, bbox])
                           
                        '''if iou >= 0.58:
                            if cont < 30:
                                h_obj[c]+=1
                                img = cv2.imread('bbox.jpg')
                                imgg=cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color =cl, thickness= 5)
                                cont+=1
                                cv2.imwrite('bbox.jpg', imgg)
                                cv2.imshow('rect', img)
                                cv2.waitKey(1000) 
                            else:
                                break'''       
            c+=1
            '''sort_sim = sorted(similar,  key=lambda x: x[0], reverse=True)
            for i in range(10):
                bbx = sort_sim[i][1]
                img = cv2.imread('bbox.jpg')
                imgg=cv2.rectangle(img,(bbx[0],bbx[1]),(bbx[2],[bbx[3]]), color =cl, thickness= 5)      
                cv2.imwrite('bbox.jpg', imgg)
                cv2.imshow('rect', img)
                cv2.waitKey(1000)'''


            
        print('nbr of objects', len(regions))       
        print('nbr anchors found to each object ')   
        print('____________________________________')    
        print(' anchors based histogram:')
        print(h_obj)    
        print(' anchors based iou:')
        print(iou_obj)
# ------------------------------- Normalized intersection similarity --------------------------
    def NI_similarity(self, h1, h2):
        sim = 0
        for a, b in zip(h1, h2):
            if a == b == 0:
                sim = sim + min(a, b) 
            else:
                t =a +b
                s = min(a,b)
                sim = s/t    
        return sim         
# ------------------------------- Generate anchor bboxes --------------------------------------
    def generate_anchors(self, point, hsimilar, dsimilar, regions):
        
        h_obj = np.zeros((len(regions)))
        d_obj = np.zeros((len(regions)))
        x = int(point[0])
        y = int(point[1]) 
        c =0
        for reg in regions: 
            gt_bbox = self.gt_bbox(reg)
            cont =0
            for i,hist in enumerate(hsimilar):
                
                print('test of the {} in dhisto:'.format(i))
                print('nbr of anchors found:', cont)
                print('______________________________________')
                pt1 = hsimilar[i][0]
                pt2 = dsimilar[i][0]
                xp1 = pt1[0]
                yp1 = pt1[1]
                
                # histogram based bbox
                xmin =int( min(x, xp1))
                ymin = int(min(y, yp1))
                xmax = int(max(x, xp1))
                ymax = int(max(y, yp1))
                bbox1 = [xmin, ymin, xmax, ymax] 
                
                iou1 = self.get_IoU(gt_bbox, bbox1)    
                if iou1 >= 0.7:
                    if cont < 10:
                        h_obj[c]+=1
                        img = cv2.imread('bbox.jpg')
                        imgg=cv2.rectangle(img,(xmin,ymin),(xmax,ymax), color =(0, 0, 0), thickness= 5)
                        cont+=1
                        cv2.imwrite('bbox.jpg', imgg)
                        cv2.imshow('rect', img)
                        cv2.waitKey(1000)  
                    else:
                        
                        break       
            for i, des in enumerate(dsimilar):
                
                print('test of the {} in descriptors:'.format(i))
                print('nbr of anchors found:', cont)
                print('______________________________________')
                # descriptor based bbox
                xp2 = pt2[0]
                yp2 = pt2[1]
                xmin1 = int(min(x, xp2))
                ymin1 = int(min(y, yp2))
                xmax1 = int(max(x, xp2))
                ymax1 = int(max(y, yp2))
                bbox2 = [xmin1, ymin1, xmax1, ymax1]
                iou2 = self.get_IoU(gt_bbox, bbox2)    
                if iou2 >= 0.7 :  
                    if cont < 10: 
                        d_obj[c]+=1
                        print('nbr anchor boxes based desc:', i)
                        img = cv2.imread('bbox.jpg')
                        imgg=cv2.rectangle(img,(xmin1,ymin1),(xmax1,ymax1), color =(255, 255, 0), thickness= 5)
                        cont+=1
                        cv2.imwrite('bbox.jpg', imgg)
                        cv2.imshow('rect', img)
                        cv2.waitKey(1000)
                    else:
                        
                        break   
               
            c+=1      
        print('nbr of objects', len(regions))       
        print('nbr anchors found to each object ')   
        print('____________________________________')    
        print(' anchors based histogram:')
        print(h_obj)
        print('**************************************************************************')
        print(' anchors based descriptors:')
        print(d_obj)  
       
 ###############################################################################################
        '''xmin = bbox_reg[0]
                ymin = bbox_reg[3]
                xmax = bbox_reg[2]
                ymax = bbox_reg[1]
                img1=cv2.rectangle(img1,(xmin,ymin),(xmax,ymax), color =(224, 23, 145), thickness= 5)
                cv2.imwrite('bbox_reg.jpg', img1)
            im = cv2.imread('bbox_reg.jpg')
            cv2.imshow('reg_bbox', im)
            cv2.waitKey(2000)'''
                
        '''cv2.drawKeypoints(img1, kp, img1, color=(255,255,0))
        cv2.imwrite('surf.jpg', img1)
        im = cv2.imread('surf.jpg')
        cv2.imshow('surf',im )
        cv2.waitKey(1000)'''
# ----------------------------------------- region extract -------------------------------------
    def region(self, img, point):
        reg = np.zeros((16,16, 3))
        x  = int(point[0])
        y  = int(point[1])
        bbox_reg = [x-8, y +7, x+7, y-8] 
        w,h = img.shape[0], img.shape[1] 
        ri =0
        for i in  range(x -8, x+8):
            rj=0
            for j in range(y-8, y+8):
                p = img[j, i]
                reg[ri, rj] = p
                rj+=1
            ri+=1
       
        return reg, bbox_reg           
# -----------------------------------------relaxed ltp ------------------------------------------        
    def relaxed_ltp8_1(self, region):
        rad = 1
        code_rltp = []
        code = [1,2,4,128,8,64,32,16]
        #gray = region[...,np.newaxis]
        w = region.shape[0]
        h = region.shape[1]
        for x_c in range(1, w-1):
            for y_c in range(1,h-1):
                p = region[x_c, y_c]
                rltp = []
                # Calculte relaxed_ltp code for all neighbors
                for i in range(x_c -rad, x_c+ rad +1):
                    for j in range(y_c-rad, y_c + rad+1):
                        z = region[i,j] -p
                        # calculate dynamic threshold based on weber's low parameter k
                        k = 0.15
                        t = round(p*k)
                        if (i, j)!= (x_c, y_c):
                            if z     >=  t: rltp.append(1)
                            if z     <= -t: rltp.append(0)
                            if abs(z)<   t: rltp.append(0.5) 
                dd = 0
                for c in range(len(code)):
                    dd = dd + rltp[c]* code[c]      
                code_rltp.append(dd) 
                region[x_c, y_c]= dd   
        hist, _ = np.histogram(region.ravel(),128,[0,127])  

        return hist      


texture = texture()
    