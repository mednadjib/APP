import sys
import numpy as np

class Dataset:
    def __init__(self):

        self.dataset = {}
        self.dataset["MSCOCO"] = {
            'cat_nbr':    80,
            'cat_name': [
                'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
                'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
                'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
                'kite','baseball bat','baseball glove','skateboard', 'surfboard','tennis racket', 
                'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
                'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
                'couch', 'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
                'keyboard','cell phone', 'microwave','oven','toaster','sink','refrigerator','book',
                'clock','vase','scissors','teddy bear','hair drier','toothbrush'],

            'cat_id':   [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
            82, 84, 85, 86, 87, 88, 89, 90 ],
            
            'mean': np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32),
            'std':  np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        }

        self.dataset["PASCAL_VOC"] = {
            'cat_nbr': 20,
            'cat_name': [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat','chair',
                'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                'sheep','sofa', 'train', 'tvmonitor'],

            'cat_id': [1, 2,3 , 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],   
            'mean' : np.array([0.39796577, 0.43019748, 0.451761 ], dtype=np.float32),
            'std':   np.array([0.06441702, 0.05864476, 0.05852588 ], dtype=np.float32)  
        } 
    
    def categories(self, dname):   
        return self.dataset[dname]['cat_nbr']
    
    def class_id(self, dname):
        class_id = {cat_id: ind for ind, cat_id in enumerate(self.dataset[dname]['cat_id'])} 
        return class_id   
    
    '''@property
    def data_config(self):
        return (configs._configs['data_config']['dataset'])

    @property
    def nbr_classes(self):
        return self.dataset[self.data_config]['cat_nbr']

    @property      
    def id_classes(self):
        return self.dataset[self.data_config]['cat_id']

    @property
    def name_classes(self):
        return self.dataset[self.data_config]['cat_name'] 
    
    @property
    def cls_dict(self):
        datas = configs._configs['data_config']['dataset'] 
        cl_dict = dict(zip(self.dataset[datas]['cat_id'], self.dataset[datas]['cat_name'])) 
        
        return cl_dict  

    @property
    def mean(self):
        return self.dataset[self.data_config]['mean']

    @property
    def std(self):
        datas = configs._configs['data_config']['dataset'] 
        return self.dataset[datas]['std']'''    


DataS = Dataset()        