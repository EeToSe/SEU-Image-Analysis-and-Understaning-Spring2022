import numpy as np

def truncated_gaussian(d, sigma = 7):
    """ Membership function of truncated Gaussian
    Input: 
        - distance from the origin 
        - sigma
    Output: 
        - degree of membership
    """
    if np.abs(d) > 2*sigma:
        return 0
    else:
        return np.exp(-np.square(d)/(2*np.square(sigma)))

def mu_black_function(z):
    """ Membership function of defining "black"
    Input:
        - crisp value: pixel intensity
    Output:
        - degree of membership that the output is black
    """
    return 0 if z > 180 else 1-z/180

def mu_white_function(z):
    """ Membership function of defining "black"
    Input:
        - crisp value: pixel intensity
    Output:
        - degree of membership that the output is black
    """
    return 0 if z < 75 else (z-75)/180

def fuzzy_filtering(n, sigma = 7):
    """ Fuzzy spatial filtering including fuzzy rule inference and defuzzification
    Input:
        - n: the pixel neighbours
        - sigma: parameter of truncated gaussian
    Output:
        - z5: Defuzzification result
    """
    distance = n - n[1][1]
    z_range = np.arange(0,256)

    # 4 neighbors' distance
    mu_zero_d2 = truncated_gaussian(distance[0][1], sigma)
    mu_zero_d4 = truncated_gaussian(distance[1][0], sigma)
    mu_zero_d6 = truncated_gaussian(distance[1][2], sigma)
    mu_zero_d8 = truncated_gaussian(distance[2][1], sigma)

    # Map a Function in NumPy With the numpy.vectorize() Function
    mu_white = np.array([mu_white_function(z5) for z5 in z_range])

    # fuzzy sets for the output
    mu1 = np.minimum(min(mu_zero_d2, mu_zero_d6), mu_white)
    mu2 = np.minimum(min(mu_zero_d6, mu_zero_d8), mu_white)
    mu3 = np.minimum(min(mu_zero_d4, mu_zero_d8), mu_white)
    mu4 = np.minimum(min(mu_zero_d2, mu_zero_d4), mu_white)
    mu5 = np.array([mu_black_function(z5) for z5 in z_range])

    mu = np.maximum.reduce([mu1, mu2, mu3, mu4, mu5])
    z5 = np.floor(sum(mu*z_range)/sum(mu))
    return z5.astype(np.uint8)