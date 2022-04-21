import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from scipy import ndimage

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
    magnitude, angle = sobel_filter(img)
    mag_thin = non_max_suppression(magnitude, angle)  
    mag_th = double_threshold(mag_thin, low_th, high_th)
    res = hysteresis(mag_th)
    return res

def sobel_filter(img, k = 3):
    '''
        Strange opencv library cv2.Sobel dont work
    '''
    # return (magnitude, angle)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    sobel_x = ndimage.filters.convolve(img, Kx)
    sobel_y = ndimage.filters.convolve(img, Ky)

    mag = np.hypot(sobel_x, sobel_y)
    mag = mag / mag.max() * 255
    theta = np.arctan2(sobel_y, sobel_x)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    return (mag, angle)    

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
    img_thin = np.zeros(img.shape)

    for i in range(1,M-1):
        for j in range(1,N-1):
            # q and r stores the intensity of neighbour pixels
            # angle 0
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
                img_thin[i,j] = img[i,j]
    return img_thin

def double_threshold(img, low_th, high_th):
    # setting the minimum and maximum thresholds 
    img_max = np.max(img)
    low_th = low_th*img_max
    high_th = high_th*img_max

    img_th = np.zeros(img.shape)
    img_th[img >= high_th] = 255
    img_th[(img < high_th) & (img > low_th)] = 75
    img_th = img_th.astype('uint8')
    return img_th

def hysteresis(img):
    height, width = img.shape
    # Looping through every pixel of the grayscale image
    img_track = img.copy()
    for i_x in range(1, width-1):
        for i_y in range(1, height-1):
            if 0 < img[i_y, i_x] < 255:
                # check 8 neighbors contain "sure-edge" pixel 
                neighbors = [img[i][j] for j in range(i_x-1, i_x+2) for i in range(i_y-1, i_y+2)]
                if max(neighbors) == 255:
                    img_track[i_y, i_x] = 255
                else:
                    img_track[i_y, i_x] = 0
    return img_track




      

       