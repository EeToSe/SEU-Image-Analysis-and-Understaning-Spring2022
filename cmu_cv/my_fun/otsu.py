from cv2 import threshold
import numpy as np

def otsu_criteria_compute(img, th):
    """
        Compute the inter-class variance for t = th

        Inputs: 
        - img: the img numpy array
        - th: a certain given threshold

        Outputs:
        - the inter-class variance for t = th
    """
    # create the threshold imagee
    img_th = np.zeros(img.shape);
    img_th[img >= th] = 1;

    # compute the class probabilities
    w0 = np.count_nonzero(img_th==0)
    w1 = 1 - w0

    # compute the class means
    mean1 = np.mean([img >= th])
    mean2 = np.mean([img < th])

    return w0*w1*((mean1-mean2)**2)

def otsu_threshold(img):
    """
        Otsu thresholding implementation.

        Inputs:
        - img: the img numpy array with uint8 data type

        Outputs:
        - a threshold 
    """
    range_th = range(np.max(img)+1)
    results = [otsu_criteria_compute(img, th) for th in range_th]
    
    threshold = range_th[np.argmax(results)]
    return threshold