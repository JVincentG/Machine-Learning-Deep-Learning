import cv2
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

def livecapture(vid):
    gray_img = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray_img, None)

    kp_img = cv2.drawKeypoints(vid, kp, None, color=(0, 255, 0), flags=0)

    cv2.imshow('ORB', kp_img)

@timer
def main(filepath):
    imgs = os.path.join(dirname, filepath)
    img = cv2.imread(imgs)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray_img, None)

    kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

    cv2.imshow('ORB', kp_img)
    cv2.waitKey()

if __name__ == "__main__":
    main()