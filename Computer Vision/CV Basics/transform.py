import cv2 as cv
import numpy as np
img = cv.imread('C:/Users/Vincent/Desktop/images/dog.jpg')



scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
cv.imshow('Original',resized)

# Translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv. warpAffine(img, transMat, dimensions)

#translated = translate(resized, 100, 100)

def rotation(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions) 
# resizing
#resized = cv.resize(resized, (500,500), interpolation=cv.INTER_CUBIC)
#cv.imshow('Resized', resized)

#Flip
flip = cv.flip(resized, -1)
#cv.imshow('Flip',flip)

#Cropping
cropped = resized[100:400, 300:400]
cv.imshow('Cropped',cropped)

#Rotating
rotated = rotation(resized, 45)
winname = 'Rotated'
#cv.namedWindow(winname)        # Create a named window
#cv.moveWindow(winname, 40,30)  # Move it to (40,30)
#cv.imshow(winname, rotated)


cv.waitKey(0)