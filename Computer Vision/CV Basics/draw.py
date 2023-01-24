import cv2 as cv
import numpy as np

blank = np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank)


""" # Paint image certain color
blank[100:400, 450:500] = 255,0,0
cv.imshow('Green', blank) """
 # Draw a rectangle

""" cv.rectangle(blank, (0,0), (250,250,), (0,0,255), thickness=5)
cv.imshow('Rectangle', blank) 

# Draw a circle
cv.circle(blank,(blank.shape[1]//2, blank.shape[0]//2),40, (0,255,0), thickness=3)
cv.imshow('circle',blank)
# Draw a line

cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness=5)
cv.imshow("Line",blank) """

cv.line(blank,(417, 1000),(417, -1000),(255,0,0),5)
cv.imshow("Line",blank)
# write text on an image
cv.putText(blank,"Hello sir Allan hehez", (0,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),1)
cv.imshow('Text',blank)
cv.waitKey(0)