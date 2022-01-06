import cv2
import random
import numpy as np 
from tqdm import tqdm
from skimage import color
from shapely.geometry import Point
from shapely.geometry import Polygon
from dataset.mscoco import coco

# --------------------------------- Region -----------------------------------------
def region(img, point):
    reg = np.zeros((16,16, 3))
    x  = int(point[0])
    y  = int(point[1])
    w,h = img.shape[0], img.shape[1] 
    ri =0
    for i in  range(x -8, x+8):
        rj=0
        for j in range(y-8, y+8):
            p = img[j, i]
            reg[ri, rj] = p
            rj+=1
        ri+=1
    return reg   
# ----------------------------------- ltp8-1 histogram -----------------------------
def ltp8_1(region): 
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
                               
# ------------------------ calcultae relaxed_ltp histogram -------------------------
def relaxed_ltp8_1(region):
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

# ------------------------  Otsu methode for threshold selection -------------------        
def Otsu(region):
# Otsu's methode for image thresholding 
# source: https://github.com/jmlipman/LAID/blob/master/IP/Otsu/otsu.py
    im_flat = np.reshape(region,(region.shape[0]*region.shape[1]))
    [hist, _] = np.histogram(region, bins=256, range=(0, 255))
    # Normalization so we have probabilities-like values (sum=1)
    hist = 1.0*hist/np.sum(hist)

    val_max = -999
    thr = -1
    for t in range(1,255):
        # Non-efficient implementation
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        m1 = np.sum(np.array([i for i in range(t)])*hist[:t])/q1
        m2 = np.sum(np.array([i for i in range(t,256)])*hist[t:])/q2
        val = q1*(1-q1)*np.power(m1-m2,2)
        if val_max < val:
            val_max = val
            thr = t
    #print("Threshold: {}".format(thr))
    return thr    