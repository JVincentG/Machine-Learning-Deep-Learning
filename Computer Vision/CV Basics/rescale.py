from turtle import width
import cv2 as cv

""" img = cv.imread('C:/Users/Vincent/Desktop/photos/dog1.png')

cv.imshow('dog',img) """

def changeRes(width,height):
    # Live Video
    capture.set(3,width)
    capture.set(4,height)


def rescaleFrame (frame,scale=0.5):
    # Existing Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

# Reading Videos
capture = cv.VideoCapture('C:/Users/ibay/Desktop/photos/dogvid.mp4')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

