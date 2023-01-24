import cv2 as cv

img = cv.imread('C:/Users/Vincent/Desktop/photos/dog.jpg')
cv.imshow('dog',img)

# Converting to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("Gray", gray)

# Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
#cv.imshow("blur", blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)
#cv.imshow("Canny Edges", canny)

# Dilating the image
dilated = cv.dilate(canny,(7,7), iterations=4)
#cv.imshow("Dilated", dilated)


# Resize
resized = cv.resize(img,(500,500))
cv.imshow('Resized',resized)

# Cropping#
cropped = img [50:200 , 200:400]
cv.imshow("Cropped",cropped)

cv.waitKey(0)