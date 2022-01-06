import numpy as np

class Config:
    
    def __init__(self):

        # Dataset parameters
        self.batch_size  = 1
        self.categories  ={"MSCOCO": 80,
                           "PASCAL_VOC": 20 }
        self._input_size = [224,224]
        

        # ROIPoolingLayer parameters:
        self.n_rois      = 1000 
        self.n_channels  = 512*3  # fixed by faster-rcnn 
        self.pooled_h    = 7
        self.pooled_w    = 7

        # model perameters 
        self.checkpointer = None
    def categories (self, dataset):
        return self.categories['dataset']
    
system_configs = Config()  
            
   
  
    

       
    


       

