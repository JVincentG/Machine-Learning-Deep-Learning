import numpy as np
import cv2 as cv
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

def harris_corner_detector(src):
    img = src
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    return img

def subpixel_accuracy(src):
    img = src
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,2,3,0.04)
    dst = cv.dilate(dst,None)
    ret, dst = cv.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]

    return img


def livecapture(vid):
    
    res = harris_corner_detector(vid)
    
    cv.imshow('Harris',res)
    # cv.imshow('dst2',res2)
    
@timer
def main(filepath):
    img = os.path.join(dirname,filepath)
    src = cv.imread(img)
    res = harris_corner_detector(src)
    res2 = subpixel_accuracy(src)
    
    cv.imshow('Harris',res)
    # cv.imshow('dst2',res2)

    cv.waitKey(0)


    
if __name__ == "__main__":
    main()