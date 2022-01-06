import sys
import os
import os.path as op
import numpy as np
import json
from  pycocotools.coco import COCO
from tqdm import tqdm
#import zip 
class coco:
# ----------------------------------   __init__   -------------------------------------------------------------------------    
    def __init__(self, src_folder, split):
        self.source_file = src_folder
        self.split = split
        self.annotations = op.join(self.source_file, 'annotations')
        self.cache_dir = op.join(self.source_file, 'cache')
        self.cache_file = op.join(self.cache_dir, 'coco_{}.json'.format(self.split))
        self.initialization()
   
    def cache(self):
        return self.cache_dir
# --------------------------------- initialisation ------------------------------------------------------------------------   
    def initialization(self):
         
        if 'trainval2014' in self.split:
            self.train_img = op.join(self.source_file, 'images', 'trainval2014')
            self.train_anno = op.join(self.annotations, 'instances_trainval2014.json')
            self.label_file = self.train_anno
            self.img_file = self.train_img
            
        if 'minival2014' in self.split:  
           
            self.val_img = op.join(self.source_file, 'images','minival2014' )
            self.val_anno = op.join(self.annotations, 'instances_minival2014.json')
            self.label_file = self.val_anno
            self.img_file = self.val_img
    
        if 'test2017'in self.split: 
            self.test_img = op.join(self.source_file, 'images','test2017' ) 
            self.test_anno = op.join(self.annotations, 'image_info_test2017.json')
            self.label_file = self.test_anno 
            self.img_file = self.test_img

# -------------------------------  load_json_file -------------------------------------------------------------------------
    def load_data(self):
           
        if not os.path.exists(self.cache_file):
            print('creating {} cache file....'.format(self.cache_file.split('/')[-1]))
            self.save_data() 
        else:  
            print('Loading  data from {}....'.format(self.label_file)) 
        with open(self.cache_file, 'r') as f:
            data, imgs = json.load(f)
        return data, imgs

# -------------------------------  save_json_file -------------------------------------------------------------------------         
    def save_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir) 
            
        self.data_extraction()
        with open(self.cache_file, 'w') as f:
            json.dump([self.images, self.detections], f, indent = 4)
# -------------------------------------------------------------------------------------------------------------------------       
    
    def data_extraction(self):
        bcat= []
        bboxes = {}
        names_img =[]
        with open(self.label_file, 'r') as f:
            data = json.load(f)
        
        img_id =  dict((img['id'], img['file_name']) for img in data['images']) 
        self.coco =COCO(self.label_file)
        for id in tqdm(img_id.keys()):
            fname = img_id[id]
            anno_id = self.coco.getAnnIds(id)
            annotations = self.coco.loadAnns(anno_id)
            path = op.join(self.img_file, fname)
            for anno in annotations:
                bcat.append([anno['bbox'][0],anno['bbox'][1],
                            anno['bbox'][2],anno['bbox'][3],
                            anno['category_id'] ])
            if len(bcat)==0:
                bboxes[path] = list([0,0,0,0,0])
                
            else:    
                bboxes[path]= bcat
              
            names_img.append(path)
                
            bcat =[]
        self.detections = bboxes      
        self.images = names_img


        
          
         

                              

       
        
            
        
       