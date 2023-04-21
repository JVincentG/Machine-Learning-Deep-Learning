import argparse
import cv2
import numpy as np
import os
import time
dirname = os.path.dirname(__file__)

#Learn more about FLANN here: https://datahacker.rs/feature-matching-methods-comparison-in-opencv/

def timer(func):
    def wrapper(*args,**kwargs):
        start = time.time()
        rv = func(*args,**kwargs)
        total = time.time() - start
        print("Time:", total)
        return rv
    return wrapper

def livecapture(vid1,vid2):
    
    res = flann(vid1,vid2)
    
    cv2.imshow('FLANN',res)


def flann(img1, img2):

 
    
    MIN_MATCHES = 50

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    for i,(m1, m2) in enumerate (matches):
        if m1.distance < 0.5 * m2.distance:
            matchesMask[i] = [1,0]
    
    draw_params = dict (matchColor = (0,0,255), singlePointColor = (0,255,0), matchesMask = matchesMask, flags=0)
    match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,**draw_params)

    return match_img


if __name__ == "__main__":
    flann()
    #img = flann('../deliverables/images/face_orig.jpg','../deliverables/images/face_tilt.jpg')
    #cv2.imshow('Corrected image', img)
    #cv2.waitKey()