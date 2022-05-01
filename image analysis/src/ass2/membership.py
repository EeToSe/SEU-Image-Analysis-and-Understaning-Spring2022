import numpy

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