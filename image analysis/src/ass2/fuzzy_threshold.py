import numpy as np

def fuzzy_threshold(frame):
    """Fuzzy thresholding implementation.
    
    Inputs:
        - img: the img numpy array with uint8 data type
    Outputs:
        - entropy for t in [0 255]
        - a threshold 
    """
    C = 256
    range_th = range(C)
    S_t = np.zeros(C, dtype=np.float64)

    for t in range(C):
        mu_x = np.zeros(frame.shape, dtype=np.float64)

        # calculate means of class intensity
        class0 = frame[frame < t]
        if (class0.size == 0):
            # deal with empty slice
            mean0 = 0
        else:
            mean0 = np.mean(frame[frame < t])
        class1 = frame[frame >= t]
        if (class1.size == 0):
            # deal with empty slice
            mean1 = 0
        else:
            mean1 = np.mean(frame[frame >= t])

        # membership function calculation
        i,j = np.where(frame < t)
        mu_x[i, j] = 1/(1+np.abs(frame[i,j]-mean0)/C)
        i,j = np.where(frame >= t)
        mu_x[i, j] = 1/(1+np.abs(frame[i,j]-mean1)/C)

        # entropy
        S_x = np.zeros(frame.shape, dtype=np.float64)
        # deal with mu_x == 1
        mu_x[mu_x == 1.0] = 0.99999
        S_x = -(mu_x*np.log(mu_x)) - (1 - mu_x)*np.log(1-mu_x)
        S_t[t] = np.sum(S_x)

    fuzzy_t = range_th[np.argmin(S_t)]
    return S_t, fuzzy_t