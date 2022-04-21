import numpy as np
import cv2

def canny_detector(img, low_th = None, high_th = None):
    '''
        Canny edge detector

        Inputs:
        - img: source image
        - weak_th: lower bound for thresholding
        - high_th: higher bound for thresholding
    '''
      
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
       
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)
      
    # Conversion of Cartesian coordinates to polar 
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
    mag_thin = non_max_suppression(mag, ang) 
    mag_th = double_threshold(mag_thin, low_th, high_th)

def non_max_suppression(mag, ang):
    '''
        Non-Maximum Suppression to thin edges:

        1. iterate pixels in gradient magnitude
        2. find maximums along the gradient orientation

        Inputs:
        - mag: gradient magnitude
        - ang: gradient orientation

        Output:
        - mag_bin： Non-Maximum Suppression of gradient magnitude
    '''
    height, width = mag.shape
    # Looping through every pixel of the grayscale image
    for i_x in range(width):
        for i_y in range(height):             
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            # compare target pixel with its neighbours
            mag_thin = mag.copy()
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag_thin[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag_thin[i_y, i_x]= 0
    return mag_thin

def double_threshold(mag, low_th, high_th):
    # setting the minimum and maximum thresholds 
    mag_max = np.max(mag)
    if not low_th:low_th = mag_max * 0.1
    if not high_th:high_th = mag_max * 0.5

    mag[mag >= high_th] = 255
    mag[(mag < high_th) & (mag > low_th)] = 128
    mag[mag <= low_th] = 0
    return mag

def hysteresis(mag):
    height, width = mag.shape
    # Looping through every pixel of the grayscale image
    for i_x in range(1, width-1):
        for i_y in range(1, height-1):
            if  0 < mag[i_y, i_x] < 255:
                # check 8 neighbors contain "sure-edge" pixel 
                neighbors = [mag[i][j] for i in range(i_x-1, i_x+2) for j in range(i_y-1, i_y+2)]
                if neighbors.max() == 255:
                    mag[i_y, i_x] = 255
                else:
                    mag[i_y, i_x] =0
    return 




      

       