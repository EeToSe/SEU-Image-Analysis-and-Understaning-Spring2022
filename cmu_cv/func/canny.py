import numpy as np
import cv2

def canny_detector(img, low_th = None, high_th = None):
    '''
        Canny edge detector

        Inputs:
        - img: grayscale source image
        - weak_th: lower bound for thresholding
        - high_th: higher bound for thresholding
    '''
    # Noise reduction step
    # img = cv2.GaussianBlur(img, (5, 5), 1.4)
        
    mag_thin = non_max_suppression(mag, ang) 
    mag_th = double_threshold(mag_thin, low_th, high_th)

def sobel_filter(img, k=3):
    # Calculating the gradients
    sobel_x = cv2.Sobel(img,-1,1,0,ksize=k)
    sobel_y = cv2.Sobel(img,-1,0,1,ksize=k)

    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees = True)
    return (255*mag/mag.max(), ang)

def non_max_suppression(img, angle):
    '''
        Non-Maximum Suppression to thin edges:

        1. iterate pixels in gradient magnitude
        2. find maximums along the gradient orientation

        Inputs:
        - img: gradient magnitude
        - ang: gradient orientation

        Output:
        - img_thin： Non-Maximum Suppression of gradient magnitude
    '''    
    M, N = img.shape
    img_thin = img.copy()
    for i in range(1,M-1):
        for j in range(1,N-1):
            # q and r stores the intensity of neighbour pixels
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                continue
            else:
                img_thin[i,j] = 0
    return img_thin

def double_threshold(img, low_th, high_th):
    # setting the minimum and maximum thresholds 
    img_max = np.max(img)
    if not low_th:low_th = img_max * 0.1
    if not high_th:high_th = img_max * 0.5
   
    img_th = img.copy()
    img_th[img_th >= high_th] = 255
    img_th[(img_th < high_th) & (img_th > low_th)] = 75
    img_th[img_th <= low_th] = 0
    img_th = img_th.astype('uint8')
    print('helloa')
    return img_th

def hysteresis(img):
    mag = img.copy()
    height, width = mag.shape

    # Looping through every pixel of the grayscale image
    for i_x in range(1, width-1):
        for i_y in range(1, height-1):
            if  0 < mag[i_y, i_x] < 255:
                # check 8 neighbors contain "sure-edge" pixel 
                neighbors = [mag[i][j] for j in range(i_x-1, i_x+2) for i in range(i_y-1, i_y+2)]
                if max(neighbors) == 255:
                    continue
                else:
                    mag[i_y, i_x] =0
    return mag




      

       