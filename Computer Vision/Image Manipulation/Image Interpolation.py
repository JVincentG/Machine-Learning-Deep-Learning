import cv2 as cv
import numpy as np
import os
dirname = os.path.dirname(__file__)

def get_histogram(src):
    bgr_planes = cv.split(src)

    histSize = 256
    histRange = (0, 256)

    accumulate = False

    b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

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
    
    img = os.path.join(dirname, '../deliverables/images/ice_cream.jpg')
    src = cv.imread(img)
    
    if src is None:
        print('Could not open or find the image')
        exit(0)

    near_img = cv.resize(src,None, fx = 10, fy = 10, interpolation = cv.INTER_NEAREST)
    bilinear_img = cv.resize(src,None, fx = 10, fy = 10, interpolation = cv.INTER_LINEAR)
    orig_img_hist = get_histogram(src)
    nn_img_hist = get_histogram(near_img)
    bi_img_hist = get_histogram(bilinear_img)
    nn_hist_eq = perform_hist_equalization(near_img)
    bi_hist_eq = perform_hist_equalization(bilinear_img)

    eq_nn_img_hist = get_histogram(nn_hist_eq)
    eq_bi_img_hist = get_histogram(bi_hist_eq)




    cv.waitKey()
if __name__ == "__main__":
    main()