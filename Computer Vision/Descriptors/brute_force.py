import cv2
import numpy as np
import os
import time
dirname = os.path.dirname(__file__)

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        rv = func(*args,**kwargs)
        total = time.time() - start
        print("Time:", total)
        return rv
    return wrapper

def livecapture(vid1,vid2):
    
    res = bfm(vid1,vid2)
    
    cv2.imshow('Brute Force',res)


def bfm(img1,img2):
    

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)

    matches = sorted(matches, key = lambda x:x.distance)
    match_img = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return match_img

if __name__ == "__main__":
    bfm()

