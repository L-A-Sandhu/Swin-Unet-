import glob
import cv2
import numpy as np
import os
path='./data/masks/*.jpg'
path1='./data/masks_png/'
if not os.path.exists(path1):
    os.makedirs(path1)
for img in glob.glob(path):
    print(img           )
    name=(img.split('/')[3]).split('.')[0]
    n= cv2.imread(img)
    #n=np.round(n/255.0)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    gray=np.round(gray/255)

    cv2.imwrite(path1+name+'.png',gray)
    