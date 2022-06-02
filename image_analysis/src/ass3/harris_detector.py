import numpy as np
from skimage import filters, io
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from skimage.feature import corner_harris, corner_peaks

def corner_detection(image, kernel_size = 3, min_distance=5, r_threshold=0.2):
    
    # Image first gradients
    Ix = cv.Sobel(image,cv.CV_64F,1,0,ksize=1)
    abs_Ix = np.absolute(Ix)
    Ix_8u = np.uint8(abs_Ix)
    Iy = cv.Sobel(image,cv.CV_64F,0,1,ksize=1)
    abs_Iy = np.absolute(Iy)
    Iy_8u = np.uint8(abs_Iy)

    # Apply Gaussian truncate window 
    sigma = 0.5
    Ixx = cv.GaussianBlur(Ix**2,(kernel_size,kernel_size), sigma)
    Ixy = cv.GaussianBlur(Ix*Iy,(kernel_size,kernel_size), sigma)
    Iyy = cv.GaussianBlur(Iy**2,(kernel_size,kernel_size), sigma)

    offset = np.int8(kernel_size/2)
    height, width = image.shape
    corner_response = np.zeros((height, width))

    # construct matrix elements to compute corner response
    k = 0.02
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            A = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            C = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            B = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            det = (A * C) - (B**2)
            trace = A + C
            r = det - k*(trace**2)
            corner_response[y][x] = r
    
    # Response threshold 0.2*r_max 
    r_max = np.max(corner_response)
    r_threshold = corner_response > 0.2*r_max
    # Non max suppression mask
    NMS = (corner_response == maximum_filter(corner_response, 5))
    mask = r_threshold & NMS
    keypoints = np.argwhere(mask==True)
    return keypoints

if __name__ == '__main__':
    image = io.imread(fname="../../data/ass3/rice.png")
    keypoints = corner_detection(image)
    
    # keypoints = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
    # plot and save
    fig, ax = plt.subplots() 
    ax.imshow(image, cmap=plt.cm.gray)
    ax.plot(keypoints[:, 1], keypoints[:, 0], color='red', marker='o',
            linestyle='None', markersize=2)
    # ax.title.set_text('Harris keypoints with non max suppresion')
    ax.set_axis_off()
    plt.show()

