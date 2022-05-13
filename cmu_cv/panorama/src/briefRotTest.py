import numpy as np
import cv2
import BRIEF

import matplotlib.pyplot as plt

def rotation_matrix(im, deg):
    ''' Calculate the rotation matrix M
    Input           Description
    ---------------------------------------------------------
    im              source image to rotate
    deg             degrees to rotate
    
    Output          Description
    ---------------------------------------------------------
    M               rotation matrix size = [2,3]
    rotated_image   im rotated by deg    
    '''
    imh, imw = im.shape[:2]
    center = (imw//2, imh//2)
    
    M = cv2.getRotationMatrix2D(center=center, angle=deg, scale=1)
    # apply affine functions to the image size as well
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((imh * sin) + (imw * cos))
    nH = int((imh * cos) + (imw * sin))

    # adjust the rotation matrix to translation take into account 
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]    
    rotated_image = cv2.warpAffine(im, M, (nW, nH))
    return M, rotated_image

def recognition_rate(im, rotated_image, M, tol=3):
    ''' Calculate the recognition rate for BRIEF descriptor
    Inputs          Description
    ---------------------------------------------------------
    im              source image to rotate
    rotated_image   im rotated by deg  
    deg             degrees to rotate
    tol             tolerance value for considering a point a correct match

    Output          Description
    ---------------------------------------------------------
    M               rotation matrix size = [2,3]
    rotated_image   im rotated by deg    
    '''
    locs1, desc1 = BRIEF.briefLite(im)
    locs2, desc2 = BRIEF.briefLite(rotated_image)
    matches = BRIEF.briefMatch(desc1, desc2)
    pt1 = locs1[matches[:,0], 0:2]
    pt2 = locs2[matches[:,1], 0:2]

    # augment pt1 to size = (3, N)
    pt1_augment = np.c_[pt1, np.ones(pt1.shape[0])].T   
    pt1_rotate = np.matmul(M, pt1_augment)

    # standard for the correct matches
    error = pt1_rotate.T - pt2
    distance = np.sqrt(error[:,0]**2+error[:,1]**2)
    matchesNum = matches.shape[0]
    result = np.ones(matchesNum, dtype=bool)
    result[distance>tol] = False
    correctNum = np.count_nonzero(result)
    return correctNum/ matchesNum

def rotate_match(im):
    """ Compare the therotical locs coordianates with the results by briefMatch 
    Input                   Description
    ---------------------------------------------------------
    im                      numpy array representing the image with size=(H, W, 3)
    
    Output                  Description
    ---------------------------------------------------------
    recognition_results     numpy array with recognition result for 
    """
    recognition_results = []
    for deg in range(0, 30, 2):
        M, rotated_image = rotation_matrix(im, deg)
        result = recognition_rate(im, rotated_image, M)
        recognition_results.append((deg, result))
    for deg in range(30, 180, 5):
        M, rotated_image = rotation_matrix(im, deg)
        result = recognition_rate(im, rotated_image, M)
        recognition_results.append((deg, result))      
    return np.array(recognition_results)

def construct_graph(recognition_results):
    """ Recurrence of paper Section4 - Orientation Sensitivity of BRIEF
    """
    x = recognition_results[:,0]
    y = 100*recognition_results[:,1]
    plt.xlabel('Rotation angle [deg]')
    plt.ylabel('recognition rate [%]')
    plt.plot(x, y,'--bo', label='BRIEF')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../data/model_chickenbroth.jpg')
    recognition_results = rotate_match(img)
    construct_graph(recognition_results)