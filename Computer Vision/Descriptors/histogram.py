import cv2 as cv
import numpy as np
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'img/p2.jpg')

def get_channels_histogram(src):
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
    
    cv.imshow('Histogram_Multiple', histImage)

def get_channel_histogram(src):
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    histSize = 256
    histRange = (0, 256)
    accumulate = False

    histg = cv.calcHist([src], [0], None, [histSize], histRange, accumulate=accumulate)

    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    cv.normalize(histg, histg, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize):
        cv.line(histImage, ( bin_w*(i-1), hist_h - int(histg[i-1]) ),
            ( bin_w*(i), hist_h - int(histg[i]) ),
            ( 255, 0, 0), thickness=2)

    cv.imshow('Histogram_Single', histImage)


def equalize_single_channel(src):
    # convert image to grayscale
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # histogram equalizer
    dst = cv.equalizeHist(src)

    return dst

def equalize_multiple_channels(src):
    ycrcb=cv.cvtColor(src,cv.COLOR_BGR2YCR_CB)
    channels=cv.split(ycrcb)
    cv.equalizeHist(channels[0],channels[0])
    cv.merge(channels,ycrcb)
    cv.cvtColor(ycrcb,cv.COLOR_YCR_CB2BGR,src)
    return src

def main():

    #read the image
    src = cv.imread('img/p2.jpg')
    
    if src is None:
        print('Could not open or find the image')
        exit(0)

    #get histogram of RGB Channels
    

    #get histogram of RBG Channels

    #show original and processed image
    # cv.imshow('Source image', src)
    # get_channel_histogram(src)

    #show processed image [SINGLE]
    img2 = equalize_multiple_channels(src)
    cv.imshow('Equalized Image', equalize_single_channel(img2))
    get_channel_histogram(img2)

    #show processed image [MUTIPLE]
    # img2 = equalize_multiple_channels(src)
    # cv.imshow('Equalized Image', img2)
    # get_channels_histogram(img2)

    cv.waitKey()

if __name__ == "__main__":
    main()


