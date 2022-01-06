import sys
import os
import os.path as op
import numpy as np
import json
from  pycocotools.coco import COCO
from tqdm import tqdm
import xmltodict
import collections
import time
class pascal_voc:
# ----------------------------------   __init__   ----------------------------------    
    def __init__(self, src_folder, split):
        self.source_file = src_folder
        self.split = split
        self.annotations = op.join(self.source_file, 'annotations')
        self.cache_dir = op.join(self.source_file, 'cache')
        self.cache_file = op.join(self.cache_dir, 'pascalvoc_{}.json'.format(self.split))
        
        self._class_name = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                   'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                   'sofa', 'train', 'tvmonitor')
        self._class = dict(zip(self._class_name, range(0, len(self._class_name) + 1)))         
        self.initialization()

# --------------------------------- initialisation ---------------------------------        
    def initialization(self):
         
        if 'trainval0712' in self.split:
            self.train_img = op.join(self.source_file, 'images', 'trainval0712')
            self.train_anno = op.join(self.annotations, 'annotations_trainval0712')
            self.label_file = self.train_anno
            self.img_file = self.train_img
            
        if 'test07' in self.split:  
           
            self.val_img = op.join(self.source_file, 'images','test07' )
            self.val_anno = op.join(self.annotations, 'annotations_test07')
            self.label_file = self.val_anno
            self.img_file = self.val_img

# -------------------------------  load_json_file ----------------------------------
    def load_data(self):
           
        if not os.path.exists(self.cache_file):
            print('creating {} cache file....'.format(self.cache_file.split('/')[-1]))
            self.save_data() 
        else:  
            print('Loading  data from {}....'.format(self.label_file)) 
        with open(self.cache_file, 'r') as f:
            data, imgs = json.load(f)
        return data, imgs

# -------------------------------  save_json_file ----------------------------------         
    def save_data(self):
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)  
        self.data_extraction()
        #self.images = os.listdir(self.img_file)  
        with open(self.cache_file, 'w') as f:
            json.dump([self.images, self.detections], f, indent = 4)   

# ---------------------  Extract regions o interest --------------------------------        
    
    def data_extraction(self, ):  
        names_img = [] 
        bboxes = {}
        print('loading annotations into memory ...')  
        xmls = os.listdir(self.label_file)
        self.detections = {}
        tic = time.time()
        for xml in tqdm(xmls):
            bbox = []
            p = os.path.join(self.label_file, xml)
            with open(p) as fd:
                anno = xmltodict.parse(fd.read(), process_namespaces=True)
                filen = anno['annotation']['filename']
                path = os.path.join(self.img_file, filen)
                obj = anno['annotation']['object']
                if type(obj)== collections.OrderedDict: obj = [obj] 
                for  i in range(len(obj)):
                    name = obj[i]['name']   
                    bb   = [obj[i]['bndbox']['xmin'],obj[i]['bndbox']['ymin'], 
                            obj[i]['bndbox']['xmax'],obj[i]['bndbox']['ymax'], self._class[name] ]
                    bbox.append(bb)

                bboxes[path]   = bbox  
                names_img.append(path)
        self.detections = bboxes
        self.images = names_img        
        print('Done (t={:0.2f}s)'.format(time.time()- tic)) 



    

    