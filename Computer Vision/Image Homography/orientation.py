from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


feature_extractor = 'sift' # one of 'sift', 'surf', 'brisk', 'orb'


def detectAndDescribe(im1Gray, im2Gray, method=None):

   assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
   
   # detect and extract features from the image
   
   if method == 'sift':
    sift = cv2.SIFT_create(MAX_FEATURES)
    kp1, d1 = sift.detectAndCompute(im1Gray, None)
    kp2, d2 = sift.detectAndCompute(im2Gray, None)

   elif method == 'surf':
    surf = cv2.xfeatures2d.SURF_create(MAX_FEATURES)
    kp1, d1 = surf.detectAndCompute(im1Gray, None)
    kp2, d2 = surf.detectAndCompute(im2Gray, None)

   elif method == 'brisk':
    brisk = cv2.BRISK_create(MAX_FEATURES)
    kp1, d1 = brisk.detectAndCompute(im1Gray, None)
    kp2, d2 = brisk.detectAndCompute(im2Gray, None)
   
   elif method == 'orb':
    orb = cv2.ORB_create(MAX_FEATURES)
    kp1, d1 = orb.detectAndCompute(im1Gray, None)
    kp2, d2 = orb.detectAndCompute(im2Gray, None)

   return kp1, kp2, d1, d2

def createMatcher(d1, d2):
  matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
  matches = matcher.match(d1, d2, None)
  return matches

def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  #2. Detect Features
  kpA, kpB, d1, d2 = detectAndDescribe(im1Gray, im2Gray, method=feature_extractor)

  fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)
  ax2.imshow(cv2.drawKeypoints(im2Gray,kpB,None,color=(0,255,0)))
  ax2.set_xlabel("Query image - Grayscale - Featured", fontsize=14)
  ax1.imshow(cv2.drawKeypoints(im1Gray,kpA,None,color=(0,255,0)))
  ax1.set_xlabel("Train image (Image to be transformed) - Grayscale - Featured", fontsize=14)

  plt.show()

  #3. Match features.
  matches = createMatcher(d1, d2)
  matches = list(matches)
  matches.sort(key=lambda x: x.distance, reverse=False)
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
  imMatches = cv2.drawMatches(im1Gray, kpA, im2Gray, kpB, matches, None)

  plt.imshow(imMatches)
  plt.show()
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = kpA[match.queryIdx].pt
    points2[i, :] = kpB[match.trainIdx].pt
  
  
  
  #4. Calculate homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)


  #5. Align image with respect to the second image
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

    
  return im1Reg, h

if __name__ == '__main__':

  #1. Read images
  origRef = imageio.imread('./Part 2/img/o_cube2.jpg')
  origim = imageio.imread('./Part 2/img/o_cube1.jpeg')

  imReference =  cv2.imread('./Part 2/img/o_cube2.jpg', cv2.IMREAD_COLOR)
  im = cv2.imread('./Part 2/img/o_cube1.jpeg', cv2.IMREAD_COLOR)

  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
  ax1.imshow(origRef, cmap="gray")
  ax1.set_xlabel("Query image", fontsize=14)

  ax2.imshow(origim, cmap="gray")
  ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)

  plt.show()

  imReg1, h = alignImages(im, imReference)

  cv2.imshow('Aligned Image', imReg1) 
  cv2.waitKey()