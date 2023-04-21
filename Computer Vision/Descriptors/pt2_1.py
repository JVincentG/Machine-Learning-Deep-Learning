import sys
import cv2 as cv
import numpy as np
import math

cv.samples.addSamplesDataSearchPath("C:\\Users\\Vincent\\Desktop\\cv\\PT2\\deliverables\\images\\")

#More details about FT is here: https://docs.opencv.org/3.4.15/d8/d01/tutorial_discrete_fourier_transform.html
def getFourierTransform(I):
    rows, cols = I.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    padded = cv.copyMakeBorder(I, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    
    planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    complexI = cv.merge(planes)         # Add to the expanded another plane with zeros
    
    cv.dft(complexI, complexI)         # this way the result may fit in the source matrix
    
    cv.split(complexI, planes)                   # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    magI = planes[0]
    
    matOfOnes = np.ones(magI.shape, dtype=magI.dtype)
    cv.add(matOfOnes, magI, magI) #  switch to logarithmic scale
    cv.log(magI, magI)
    
    magI_rows, magI_cols = magI.shape
    # crop the spectrum, if it has an odd number of rows or columns
    magI = magI[0:(magI_rows & -2), 0:(magI_cols & -2)]
    cx = int(magI_rows/2)
    cy = int(magI_cols/2)
    q0 = magI[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
    q1 = magI[cx:cx+cx, 0:cy]     # Top-Right
    q2 = magI[0:cx, cy:cy+cy]     # Bottom-Left
    q3 = magI[cx:cx+cx, cy:cy+cy] # Bottom-Right
    tmp = np.copy(q0)               # swap quadrants (Top-Left with Bottom-Right)
    magI[0:cx, 0:cy] = q3
    magI[cx:cx + cx, cy:cy + cy] = tmp
    tmp = np.copy(q1)               # swap quadrant (Top-Right with Bottom-Left)
    magI[cx:cx + cx, 0:cy] = q2
    magI[0:cx, cy:cy + cy] = tmp
    
    cv.normalize(magI, magI, 0, 1, cv.NORM_MINMAX) # Transform the matrix with float values into a
                                                   # viewable image form (float between values 0 and 1). 
    return magI

def getLine(img, img2):
    temp = img
    temp *= 255/temp.max() 
    temp = temp.astype(np.uint8)
    temp = 255 - temp
    thresh = cv.adaptiveThreshold(temp, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 3)
    thresh = 255 - thresh

    # apply close to connect the white areas
    kernel = np.ones((3,3), np.uint8)
    morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    kernel = np.ones((1,9), np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

    # apply canny edge detection
    edges = cv.Canny(morph, 150, 200)

    # get hough lines
    result = img.copy()
    lines = cv.HoughLines(edges, 1, np.pi/180, 50)

    # Draw line on the image
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        
        cv.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return [x1,y1,x2,y2], img2

def get_and_draw_Angle(line, img):
    # HINT: Review your geometry
    try:
        m = (line[3]-line[1])/(line[2]-line[0])
    except:
        m = 0
    angle_in_radians = math.atan(m) 
    angle_in_degrees = math.degrees(angle_in_radians)
    txt = cv.putText(img,"Angle: "+ str(angle_in_degrees), (0,30),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),1)
    return txt , angle_in_degrees


def rotate_image(image, angle):
   # HINT: Review geometric transformation topic
   if angle != 0:
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rotation_value = 270 + angle
    M = cv.getRotationMatrix2D((cX, cY), rotation_value, 1.0)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated
   txt = cv.putText(image,"Image is not tilted", (0,90),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),1)
   return image

def main(argv):
    #Some of the functions are provided, you just need to fill in the missing parts. 
    
    #Here's a guide for you: 
    #1. Read the Inputs
    filename = argv[0] if len(argv) > 0 else 'pt1_original.png'
    I = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if I is None:
        print('Error opening image')
        return -1
    
    scale_percent = 50 # percent of original size
    width = int(I.shape[1] * scale_percent / 100)
    height = int(I.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    I = cv.resize(I, dim, interpolation = cv.INTER_AREA)

    #2. Get the inputs magnitude spectrums
    mag_spectrum = getFourierTransform(I)

    #3. Display the magnitude spectrum of the tilted image
    #cv.imshow("Input Image"       , I   )    # Show the result
    cv.imshow("spectrum magnitude", mag_spectrum)
    #cv.waitKey()

    #4. Draw a line and write a text (of the angle value) on the titled image for reference
    # write text on an image
    gl = getLine(mag_spectrum, I)
    line_img = gl[1]
    GDA = get_and_draw_Angle(gl[0], line_img)
    GDA[0]
    #5. Display the image with the line and written text
    cv.imshow("Input Image", line_img)

    #6. [OPTIONAL] Correct the image
    cv.imshow("Rotated Image",rotate_image(I, GDA[1]))


    cv.waitKey()


if __name__ == "__main__":
    main(sys.argv[1:])