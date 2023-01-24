import cv2 as cv

#img = cv.imread('C:/Users/Vincent/Desktop/photos/cat1.png')

#cv.imshow('cat',img)
# Reading Videos
capture = cv.VideoCapture('C:/Users/Vincent/Desktop/photos/catvid.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()