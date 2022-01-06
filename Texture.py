import os
import sys
import cv2
import random
import numpy as np 
import itertools
from tqdm import tqdm
from skimage import color
from shapely.geometry import Point
from shapely.geometry import Polygon

import dataset.dataset_select as DS
import json
from ROI.outil import region, ltp8_1, relaxed_ltp8_1
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import os
import cv2
os.environ['DISPLAY'] = ':0' # Solution of "cannot connect to X server" whene excute cv2.imshow()


class texture:
    def __init__(self):

        src_folder   = sys.argv[1]
        dataset_name = sys.argv[2]
        subset       = sys.argv[3].split(',')
        ind          = int(sys.argv[4])
        
        print('Processing Image: ', ind)
        if dataset_name == 'MSCOCO':
            train_subset = subset[0]
            val_subset   = subset[1]
            test_subset  = subset[2]

        if dataset_name == 'PASCAL_VOC':
            train_subset = subset.split(',')[0]
            val_subset  = subset.split(',')[1]
            path = ''
            regions = ''
       
        train_data = DS.datasets[dataset_name](src_folder,train_subset).load_data()
        val_data = DS.datasets[dataset_name](src_folder,val_subset).load_data()
        
        if ind ==0:
            print('Dataset name:   {}\n'.format(dataset_name))
            print('Loading training data ..........\n')
            print('--> {} items found   in training subset \n '.format(len(train_data[0])))
            print('--> {} items found in validation subset \n '.format(len(val_data[0])))
        
        # Images/regions paths
        self.paths = list(train_data[1].keys())
        self.rois  = list(train_data[1][path] for path in self.paths)

        # Create anchors folder
        self.anchors_dir = os.path.join(src_folder, 'cache/{}_anchors1'.format(dataset_name))
        
        if ind <= len(train_data[0]):

            img_path = self.paths[ind]
            
            regions  = self.rois[ind]
       
            self.name_img = img_path.split('/')[-1]
            self.regions = regions
            self.img = cv2.imread(img_path) 
            
            anchors = self.process()

            # Save anchors in json file
        
            if len(anchors[0])== 0 &  len(anchors[1])== 0:
                self.anchors_file = os.path.join(self.anchors_dir, 'None_{}.json'.format(ind))
            else:
                self.anchors_file = os.path.join(self.anchors_dir, 'anch_{}.json'.format(ind))    
    
            self.save_anchors(self.paths[ind], anchors)
        else:
            print('OUTSIDE DATASET')   
           
# -------------------------------------------------------------------------------------------------      
    def process(self):
       
        self.anchors_dict = {}
        foreground = []
        background = []
        h, w = self.img.shape[1], self.img.shape[0] 
        
        if self.regions == [0,0,0,0,0]: 
            selected_pt= []
            print('no ground truth annotations...')
         
        else: 
            print('\n 1-  keypoints detection......')   
            selected_pt = self.Kpoint_detection()
            
            gt_boxes = []
            # Extract ground truth boxes
            for region in self.regions:
                b = self.gt_bbox(region)
                gt_boxes.append(b)

            print('\n 2- keypoints anchors generation......')
            
            anchors      = self.anchor_gen2(selected_pt, self.img) 
            
            c = len(anchors)
            print( ' \n', c,    'anchors found')
            #anchors/ regions identities
            dict_anchors = {id:anchor for id, anchor in enumerate(anchors)}
            dict_regions = {id: region for id, region in enumerate(gt_boxes)}
            
            # calculate iou
            matrix_iou   = self.get_matrix_iou(anchors, gt_boxes)
            
            print('\n 3- Labels assignement ...')
            fg_anchors, bg_anchors = self.assignement(matrix_iou, dict_anchors)    

            print('\n 4 - Prepare anchors for training phase......')
            
            foreground, background = self.prepare_anchors( fg_anchors, bg_anchors)
       
        return [foreground, background]
    
    # --------------------------------------------------------------------------------------------------------------------- 
    def assignement(self, matrix_iou, dict_anchors):

        # -----------------------------------------------------------------------------------------------------------------
        # ASSIGNE LABELS TO ANCHORS (1: FG, 0: BG, -1: OTHERWISE)
            
        # case1: assigne  1  to anchors that have  maximum iou  with grondtruth boxes  # we must have at least 1 fg anchor
        # case2: assigne  1  to anchors with iou >= 0.7
        # case3: assigne  0  to anchors with iou < 0.3
        # case4: assigne -1  otherwise
        # -----------------------------------------------------------------------------------------------------------------
        anch_labels = []
        # case1
        index_max = []
        for row in tqdm(matrix_iou): 
            max_iou = max(row)
            index   = row.index(max_iou)
            anch    = dict_anchors[index]
            anch_labels.append([anch[0], anch[1], anch[2], anch[3], 1])
                
            index_max.append(index)   
            
        for row in tqdm(matrix_iou):
            reg_ind = matrix_iou.index(row)
                
            for iou in row:
                index_iou = row.index(iou)
                if index_iou not in index_max:       # to avoid to duplicate  the 1st case labels
                    index_anchor = row.index(iou)
                    anch = dict_anchors[index_anchor]
                      
                    #case 2
                    if iou >=  0.7:
                        
                        anch_labels.append([anch[0], anch[1], anch[2], anch[3], 1])
                    
                    # case 3
                    elif iou <= 0.3:
                            
                        anch_labels.append([anch[0], anch[1], anch[2], anch[3], 0])
                            
                    # case 4
                    else:
                        anch_labels.append([anch[0], anch[1], anch[2], anch[3], -1])    
    
            
        fg_anchors = []
        bg_anchors = []
        for label in anch_labels:
            if label[-1] == 1:
                fg_anchors.append(label)
                    
            if label[-1] == 0:
                    
                bg_anchors.append(label)    
                           
            
        fg_anchors.sort()
        fg_anchors = list(k for k,_ in itertools.groupby(fg_anchors))  # Remove redundant label fg anchors

        bg_anchors.sort()
        bg_anchors = list(k for k,_ in itertools.groupby(bg_anchors))  # Remove redundant label bg anchors
            
        print('_____________________________________________________________________')
        print('{} fg anchors:'.format( len(fg_anchors)), '------- {} bg anchors:'.format( len(bg_anchors)) )
        print('_____________________________________________________________________')
        
        return fg_anchors, bg_anchors
    
    # ---------------------------------------------------------------------------------------------------------------------
    def prepare_anchors(self, fg_anchors, bg_anchors):       

        # -----------------------------------------------------------------------------------------------------------------
        #                            PREPRARE ANCHORS FOR TRAINNIG PHASE:
        # -----------------------------------------------------------------------------------------------------------------
            
        # 1-ANCHORS SELECTION: 
        #     select 128 forground anchors   
        #     select 128 background anchors 
        #     if  number of detected forgroud anchors < 128  then: add background anchors
        # 2- TRANSFORM ANCHORS TO TRAINING FORM: [xmin, ymin, width, height]
        # -----------------------------------------------------------------------------------------------------------------
        fg = []
        bg = []
        
        if len(fg_anchors)< 128:
            fg   = fg_anchors
            diff = 128 - len(fg_anchors)
            for i in range(diff):
                fg.append(bg_anchors[i])
                bg_anchors.pop(i)
        else:
            fg = random.sample(fg_anchors, 128)   
        bg = random.sample(bg_anchors, 128)     
        #__________________________________________________________________________________________________________________
        '''colors = [(random.randint(0, 255),random.randint(0, 255), random.randint(0, 255) ) for _ in range(len(fg))]
        cl = random.choice(colors)
        im2 = self.img.copy()
        for anch in fg  :
            xmin = anch[0]
            ymin = anch[1]
            xmax = anch[2]
            ymax = anch[3]
            
            imgg=cv2.rectangle(im2,(xmin,ymin),(xmax,ymax), color =cl , thickness= 3)
            cv2.imwrite('fg_{}.jpg'.format(self.name_img), imgg)
        #__________________________________________________________________________________________________________________
        colors = [(random.randint(0, 255),random.randint(0, 255), random.randint(0, 255) ) for _ in range(len(bg))]
        cl = random.choice(colors)
        im3 = self.img.copy()
        for anch in bg  :
            xmin = anch[0]
            ymin = anch[1]
            xmax = anch[2]
            ymax = anch[3]
            
            imgg=cv2.rectangle(im3,(xmin,ymin),(xmax,ymax), color =cl , thickness= 5)
        #__________________________________________________________________________________________________________________    
            cv2.imwrite('bg_{}.jpg'.format(self.name_img), imgg)'''
        # Convert to [xmin, ymin, widht, height]  format 
        fg_anch = []
        bg_anch = []
        fg_anch = self.convert_format(fg)
        bg_anch = self.convert_format(bg)
        
        return fg_anch, bg_anch
    # ---------------------------------------------------------------------------------------------------------------------
    def convert_format(self, lst_anch):
        conv_lst = []   
        
        for i in range(len(lst_anch)):
            xmin  = lst_anch[i][0]
            ymin  = lst_anch[i][1]
            xmax  = lst_anch[i][2]
            ymax  = lst_anch[i][3]
            label = lst_anch[i][4]
            w     = xmax - xmin
            h     = ymax - ymin
            conv_lst.append([xmin, ymin, w, h, label])
        
        return conv_lst    
      
# -------------------------------------------------------------------------------------------------------------------------    
    def  get_matrix_iou(self, anchors, regions):
        
        matrix_iou = []
        for region in tqdm(regions):
            reg_iou = []
            for id, anchor in enumerate(anchors):
                
                iou = self.compute_iou(anchor, region)
                reg_iou.append(iou)
            matrix_iou.append(reg_iou)      
        
        return matrix_iou
# -------------------------------------------------- Kpoint_detection -----------------------------------------------------
 
    def Kpoint_detection(self):
        
        img1 = cv2.GaussianBlur(self.img,(3,3),0)
        #cv2.imwrite('org_{}.jpg'.format(self.name_img),img1)
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(0)
        kp1, des1 = surf.detectAndCompute(img1, None)
           
        sift = cv2.xfeatures2d.SIFT_create()
        kp2, des2 = sift.detectAndCompute(img1, None)     
        
        kp = kp1 + kp2    
        '''im = cv2.imread('org_{}.jpg'.format(self.name_img))
        cv2.drawKeypoints(im, kp, im, color=(0,255,0))
        cv2.imwrite('surf_{}.jpg'.format(self.name_img), im)'''
        
        kpts = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave,k.class_id]
                            for k in kp])  
        kpd = []
        for point in kpts:
            
            kpd.append([point[0], point[1]])             
        
        # sort points based on espence value                
        sorted_pt   = sorted(kpts,  key=lambda x: x[4], reverse=True)  
        selected_pt = [[0, 0] for i in range(400)]
        if len(sorted_pt)>= 400: 

            for i in range(len(selected_pt)):
               
                selected_pt[i] = [sorted_pt[i][0], sorted_pt[i][1]]
        else:
            for i in range(len(sorted_pt)) :
                
                selected_pt[i] = [sorted_pt[i][0], sorted_pt[i][1]]

            
        print(' {} keypoints selected'.format(len(selected_pt)))
        
        return selected_pt           
# ----------------------------------------------------------------------------------------------
    def anchor_gen2(self, keypoints, img):
        
        c = 0
        
        bboxes = [] 
        keypoints2 = keypoints
        random.shuffle(keypoints2)
        
        for pt in tqdm(keypoints):
            
            x = pt[0]
            y = pt[1]
            for pt1 in keypoints2: 
                
                if pt != pt1:
                    xp = pt1[0]
                    yp = pt1[1]
                    xmin =int( min(x, xp))
                    ymin = int(min(y, yp))
                    xmax = int(max(x, xp))
                    ymax = int(max(y, yp))
                    bbox = [xmin, ymin, xmax, ymax] 
                   
                    if (xmax > xmin) and  (ymax > ymin):   # discard small ereas 
                    #if (xmax !=  xmin)or (ymax != ymin):
                        bboxes.append(bbox)  
                        #c =c+1
                        
                                       
        bboxes.sort()
        anchors = list(k for k,_ in itertools.groupby(bboxes))   # delete duplicate anchors   
        '''colors = [(random.randint(0, 255),random.randint(0, 255), random.randint(0, 255) ) for _ in range(len(anchors))] 
        im2 = img.copy() 
        for anch in tqdm(anchors):
            cl = random.choice(colors)
            xmin = anch[0]
            ymin = anch[1]
            xmax = anch[2]
            ymax = anch[3]
            
            imgg=cv2.rectangle(im2,(xmin,ymin),(xmax,ymax), color =cl , thickness= 3)
            cv2.imwrite('bbox_{}.jpg'.format(self.name_img), imgg)'''

        return anchors            

# ----------------------------------- extract ground truth bbox -----------------------------------------------------------
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
      
# ------------------------------------------------------------------------------------------------------------------------- 
    def save_anchors(self, img_path, anchors):
        if not os.path.exists(self.anchors_dir):
            os.mkdir(self.anchors_dir) 
        
        anchors_dict = {}
        anchors_dict[img_path]= anchors
        
        with open(self.anchors_file, 'w') as f:
            json.dump(anchors_dict, f, indent = 4)        

# -------------------------------------------------------------------------------------------------------------------------

# ---------------------------------- calculate intersection over union between tow regions --------------------------------
    def get_IoU(self, bb1, bb2):
        x_left   = max(bb1[0], bb2[0])
        y_top    = max(bb1[1], bb2[1])
        x_right  = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        bb1_area = abs((bb1[2]- bb1[0]))*abs((bb1[3]- bb1[1]))
        bb2_area = abs((bb2[2]- bb2[0]))*abs((bb2[3]- bb2[1]))
        
        #intersect_area = abs(max((x_right - x_left), 0) * max((y_bottom - y_top),0))
        intersect_area = abs((x_right - x_left)) * abs((y_bottom - y_top))
        iou = intersect_area / float(bb1_area + bb2_area - intersect_area)
        
        return iou
# ------------------------------------------- Normalized intersection similarity ------------------------------------------
    def compute_iou(self, box1, box2):
        
        '''imgg=cv2.rectangle(self.img,(box1[0],box1[1]),(box1[2],box1[3]), color =(100, 100, 100) , thickness= 5)
        name_img = 'iou_{}.jpg'.format(c)
        cv2.imwrite(name_img, imgg)

        imgg=cv2.rectangle(self.img,(int(box2[0]),int(box2[1])),(int(box2[2]),int(box2[3])), color =(100, 100, 100) , thickness= 10)
        cv2.imwrite(name_img, imgg
        '''
        x_min_inter = max(box1[0], box2[0])
        x_max_inter = min(box1[2], box2[2])
        y_min_inter = max(box1[1], box2[1])
        y_max_inter = min(box1[3], box2[3])
        '''imgg=cv2.rectangle(self.img,(int(x_min_inter),int(y_min_inter)),(int(x_max_inter),int(y_max_inter)), color =(255, 255, 250) , thickness= 5)
        cv2.imwrite(name_img, imgg)

        img = cv2.imread(name_img)
        cv2.imshow('iou', img)
        cv2.waitKey(10000)'''
        
        # box1 area
        h_box1    = abs(box1[2] - box1[0])
        w_box1    = abs(box1[3] - box1[1])
        area_box1 = (h_box1 * w_box1)

        # box2 area
        h_box2    = abs(box2[2] - box2[0])
        w_box2    = abs(box2[3] - box2[1])
        area_box2 =( h_box2 * w_box2)
        
        # area intersection
        h_inter   =  max((x_max_inter - x_min_inter), 0)
        w_inter    = max((y_max_inter - y_min_inter), 0)
        area_inter = (h_inter * w_inter)
        #area union
        area_union = (area_box1 + area_box2 - area_inter)
        '''print(x_max_inter, '____min ____',box1[2], box2[2])
        print(x_min_inter, '____max____', box1[0], box2[0])
        print(y_max_inter, '_____min___', box1[3], box2[3])
        print(y_min_inter, '___max_____',box1[1], box2[1])
        print('---------------------------------------------')'''
        
        # compute iou 

        iou = (area_inter/ area_union)
        
        '''box_1 = [[box1[0], box1[1]], [box1[2], box1[1]],[box1[2], box1[3]],  [box1[0], box1[3]]]
        box_2 = [[box2[0], box2[1]], [box2[2], box2[1]],[box2[2], box2[3]],  [box2[0], box2[3]]]
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
    
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area'''
    
        return iou 



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
# ------------------------------------- ground truth bbox -----------------------------------------

    def draw_gtbbox (self,image,  regions):
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
         

# ----------------------------------- ltp8-1 histogram --------------------------------------------
    def ltp8_1(self, region): 
        code = [1,2,4,128,8,64,32,16]
        reg_p = np.copy(region)
        reg_n = np.copy(region)
        w = region.shape[0]
        h = region.shape[1]
        for x_c in range(1, w-1):
            for y_c in range(1, h-1):
                c = region[x_c, y_c]
                cltp =[]
                for i in range(x_c-1, x_c+2):
                    for j in range(y_c-1, y_c+2):
                        z = region[i, j] -c
                        # t is calculated automaticly based on weber's low parameter
                        t = round(c*0.15)
                        if (i, j)!= (x_c, y_c):
                            if z     >=  t: cltp.append(1)
                            if z     <= -t: cltp.append(0)
                            if abs(z)<   t: cltp.append(-1)           
                cltp_pos = []
                cltp_neg = []
                # generate positive and negative codes
                for i in range(len(cltp)):
                    if cltp[i] == -1:
                        cltp_pos.append(0)  
                        cltp_neg.append(1)  
                    elif cltp[i]== 1: 
                        cltp_neg.append(0)
                        cltp_pos.append(cltp[i])
                    elif cltp[i] == 0:    
                        cltp_neg.append(cltp[i])
                        cltp_pos.append(cltp[i])  
                d1 = 0
                d2 = 0
                for i, j in zip(range(len(cltp_pos)), range(len(cltp_neg))):
                    d1 = d1 +cltp_pos[i]*code[i] 
                    d2 = d2 +cltp_neg[j]*code[j] 
                reg_p[x_c, y_c] = d1 
                reg_n[x_c, y_c] = d2      
        histp, _ = np.histogram(reg_p.ravel(),128,[0,127])
        histn, _ = np.histogram(reg_n.ravel(),128,[0,127])
        return histp, histn
                               
# -------------------------------- calcultae relaxed_ltp histogram -------------------------------
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
# ---------------------------------------- extract region ----------------------------------------
    def reg_image(self, img, rois): 
        for obj in rois:
            xmin = obj[0]
            ymin = obj[1]
            w   = obj[2]
            h   = obj[3]
            if 'COCO' not in self.name_img: 
                xmax = int(w)
                ymax = int(h)
            else:    
                xmax = xmin + int(w)
                ymax = ymin + int(h)
            regg = img[int(ymin):int(ymax), int(xmin):int(xmax)]   
            cv2.imwrite('reg_extract.jpg', regg)
            re = cv2.imread('reg_extract.jpg')
            cv2.imshow('regg',re )
            cv2.waitKey(2000)

