import cv2 as cv
import numpy as np
import os
dirname = os.path.dirname(__file__)

#To learn more about interpolation, visit this link: https://theailearner.com/2018/11/15/image-interpolation-using-opencv-python/

def get_histogram(src):
    #histogram
    #splitting the source image to RGB planes
    bgr_planes = cv.split(src)

    #set the range of values. RGB has 0 to 255
    histSize = 256
    histRange = (0, 256)

    #Assuring that histograms are cleared at the beginning
    accumulate = False

    #calculate the histogram
    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    #image specification
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    #normalize
    cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(b_hist[i]) ),
                ( 255, 0, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(g_hist[i]) ),
                ( 0, 255, 0), thickness=2)
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
                ( bin_w*(i), hist_h - int(r_hist[i]) ),
                ( 0, 0, 255), thickness=2)
    
    return histImage

def perform_hist_equalization(src):
    ycrcb=cv.cvtColor(src,cv.COLOR_BGR2YCR_CB)
    channels=cv.split(ycrcb)
    cv.equalizeHist(channels[0],channels[0])
    cv.merge(channels,ycrcb)
    cv.cvtColor(ycrcb,cv.COLOR_YCR_CB2BGR,src)
    return src

def main():
    
    #Some of the functions are provided, you just need to fill in the missing parts. 

    #Here's a guide for you: 
    #1. Read the Inputs
    img = os.path.join(dirname, '../deliverables/images/ice_cream.jpg')
    src = cv.imread(img)
    
    if src is None:
        print('Could not open or find the image')
        exit(0)

    #2. Interpolate each image with a scaling factor fx = 10, and fy = 10 using nearest neighbor and bilinear interpolation
    near_img = cv.resize(src,None, fx = 10, fy = 10, interpolation = cv.INTER_NEAREST)
    bilinear_img = cv.resize(src,None, fx = 10, fy = 10, interpolation = cv.INTER_LINEAR)


    #3. Generate and display the histogram of the original image, and the two ouput images
        # original image
    #cv.imshow('Original Image', src)
    orig_img_hist = get_histogram(src)
    #cv.imshow('Original Image Histogram', orig_img_hist)

        # nearest neighbor
    #cv.imshow('Nearest Neighbor Image', near_img)
    nn_img_hist = get_histogram(near_img)
    #cv.imshow('Nearest Neighbor Histogram', nn_img_hist)

        # bilinear interpolation
    #cv.imshow('Bilinear Image', bilinear_img)
    bi_img_hist = get_histogram(bilinear_img)
    #cv.imshow('Bilinear Histogram', bi_img_hist)
    

    #4. Perform Histogram equalization for both output images. Refer to histogram.py for the use of the functions above
    nn_hist_eq = perform_hist_equalization(near_img)
    bi_hist_eq = perform_hist_equalization(bilinear_img)


    #5. Generate and display the histogram of each equalized output image (i.e. generated image from nearest neighbor and bilinear interpolation).
    eq_nn_img_hist = get_histogram(nn_hist_eq)
    eq_bi_img_hist = get_histogram(bi_hist_eq)

    
    #cv.imshow('Nearest Neighbor Equalized Image',nn_hist_eq)
    #cv.imshow('Bilinear Equalized Image',bi_hist_eq)


    cv.waitKey()
if __name__ == "__main__":
    main()