import sys
import os 
import numpy as np 
import argparse

from tqdm import tqdm
import dataset.dataset_select as DS
from ROI import test_mod
from ROI.mod import MODEL
#from ROI import Texture as tex
from ROI.Texture import texture 
#from parallel import syncronize
from Model import  network
import subprocess
import cv2
import json

#GLOBAL VARIABLES


# -------------------------------------------------------------------------------------------------------------------------
def parse_args():
    
    parser = argparse.ArgumentParser(description = 'Training convolutional network using local texture descriptors ')
    parser.add_argument('--src_file', dest= 'src_file', default='data/coco', help = 'Source folder')
    parser.add_argument('--dataset', dest='dataset', default='MSCOCO', help='dataset used for training')
    parser.add_argument('--split', dest='split', help = 'train images folder,validation images folder, test images folder')
    #parser.add_argument('--test', dest= 'test', default='data/test', help= 'The folder with testing data')
    # continue...
    pars = parser.parse_args()
    return pars
# -------------------------------------------------------------------------------------------------------------------------
def train(train_data, val_data, dataset):

    train_size = len([k for k in train_data.keys()])
    val_size   = len([k for k in val_data.keys()])
    
    model = MODEL(train_data, train_size, dataset_name)
    model.train()

# --------------------------------------------------------------------------------------------------------------------------
def syncronize(p):
   for v in main():
       print(v)
# --------------------------------------------------------------------------------------------------------------------------
        
def main():
    pars = parse_args()
       
if __name__ == '__main__':   
    pars = parse_args()
    src_folder = pars.src_file
   
    dataset_name = pars.dataset
    subset = pars.split
    '''if dataset_name == 'MSCOCO':
        train_subset = subset.split(',')[0]
        val_subset   = subset.split(',')[1]
        test_subset  = subset.split(',')[2]

    if dataset_name == 'PASCAL_VOC':
        train_subset = subset.split(',')[0]
        val_subset  = subset.split(',')[1]
    
    #sys.exit()
    #Loading trainig data
    print('Dataset name:   {}'.format(dataset_name))
    print('Loading training data ..........')
    
    train_data = DS.datasets[dataset_name](src_folder,train_subset).load_data()
    
    train_cache_file = DS.datasets[dataset_name](src_folder,train_subset).cache_file
    
    print('--> {} items found   in training subset \n '.format(len(train_data[0])))
    
    val_data = DS.datasets[dataset_name](src_folder,val_subset).load_data()
    print('--> {} items found in validation subset \n '.format(len(val_data[0])))
  
    
    paths = list(train_data[1].keys())'''
    texture(src_folder, dataset_name, subset, 892 )
    '''
    sync = 5
    k_ind = 177
   
    rnd = 1
    while k_ind <= 184:
        list_indx = []
        paths_batch = []
        print('______________________________________________________________________________________')
        print("Batch: {}  .....".format(rnd), 'count images: {} '.format(sync))
        for i in range(sync):
            list_indx.append(k_ind)
            print('Image: {}'.format(k_ind), paths[k_ind].split('/')[-1]) 
            k_ind+=1
            if k_ind > 184:
                break
       
        rnd+=1
        subprocess.Popen(['./launch.sh', str(train_cache_file)] + [str(n) for n in list_indx], shell=False)  
          
       '''
        
        
        
        
        
        
        
        
        
        

        

    '''sync(paths, rois)
    sys.exit()  
    for path in tqdm(train_data[1].keys()):
        print('anchors of', nb_img, path )
        print('____________________________________________________________')
        nb_img = nb_img + 1
        #path = dict_path[121]    
        rois = train_data[1][path]
        anchors_bias = texture(path, rois).process()
        anchors_dict[path]= anchors_bias
        if not os.path.exists(anchor_dir):
            os.mkdir(anchor_dir)
        with open(anchor_file, 'w') as f:
            json.dump(anchors_dict, f, indent= 4)
'''
    #sys.exit()
    #train(train_data[1], val_data[1], dataset_name)
    
   
    
     

    
   
   
    

   
    
    