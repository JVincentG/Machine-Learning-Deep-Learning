# Code Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
 
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
dirname = os.path.dirname(__file__)
import time

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        rv = func(*args,**kwargs)
        total = time.time() - start
        print("Time:", total)
        return rv
    return wrapper

def livecapture(vid):

    gray_img = cv.cvtColor(vid, cv.COLOR_BGR2GRAY)
    
    # Find the top 20 corners
    corners = cv.goodFeaturesToTrack(gray_img,20,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv.circle(vid,(x,y),3,255,-1)
    
    cv.imshow('Shi-Tomasi', vid)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

@timer   
def main(filepath):
    imgs = os.path.join(dirname, filepath)
    img = cv.imread(imgs)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    # Find the top 20 corners
    corners = cv.goodFeaturesToTrack(gray,20,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    
    cv.imshow('Shi-Tomasi', img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

if __name__ == "__main__":
    main()