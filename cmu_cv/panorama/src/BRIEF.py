import numpy as np
import cv2
import os
import glob
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    ''' Spatial Arrangement of the BRIEF Tests
    Inputs          Description
    --------------------------------------------------------------------------
    patch_width     the width of the image patch
    nbits           the number of tests n in the BRIEF descriptor

    Outputs         Description
    --------------------------------------------------------------------------
    compareX &      LINEAR indices into the flatten patch_width x patch_width image
    compareY        of size = (nbits, 1) vectors each for compareX and compareY 
    '''
    # Spatial Arrangement of the Binary Tests - I) Uniform
    # load test pattern for Brief
    test_pattern_file = '../results/testPattern.npy'

    if os.path.isfile(test_pattern_file):
        # load from file if exists
        compareX, compareY = np.load(test_pattern_file)
    else:
        # produce and save patterns if not exist
        compareX = np.random.randint(patch_width**2, size=(nbits, 1), dtype=int)
        compareY = np.random.randint(patch_width**2, size=(nbits, 1), dtype=int)
        if not os.path.isdir('../results'):
            os.mkdir('../results')
        np.save(test_pattern_file, [compareX, compareY])
    return compareX, compareY

def computeBrief(im, locsDoG, gaussian_pyramid, compareX, compareY):
    ''' Compute Brief feature with the set of BRIEF Tests 
    Inputs          Description
    --------------------------------------------------------------------------
    im              grayscale image with only 1 channel
    locsDoG         (N, 3) matrix where the DoG pyramid achieves a local extrema in both
                    scale and space, and also satisfies the two thresholds.
    levels          Gaussian scale levels that were given in Section1.
    compareX &      LINEAR indices into the flatten patch_width x patch_width image
    compareY        of size = (nbits, 1) vectors each for compareX and compareY 

    Outputs         Description
    --------------------------------------------------------------------------
    locs            an m x 3 vector, where the first two columns are the image
    		        coordinates of keypoints and the third column is the pyramid
                    level of the keypoints.
    desc            an m x nbits matrix of stacked BRIEF descriptors.
                    m: # of valid descriptors in the image
    '''
    # conner case: keypoint at the edge of the image -> no patch of width patch_width
    EXTENT = 4
    locs = []
    imH, imW = im.shape
    for i in range(locsDoG.shape[0]):
        loc = locsDoG[i,:]
        h, w = loc[1], loc[0]
        if h+EXTENT < imH and h-EXTENT >= 0 and w+EXTENT < imW and w-EXTENT >= 0:
            locs.append(loc)

    # compute BRIEF descriptor
    nbits = compareX.size
    desc = []
    for loc in locs:
        test =[]
        h, w = loc[1], loc[0]
        patch = im[h-EXTENT:h+EXTENT+1, w-EXTENT:w+EXTENT+1]
        patch = patch.flatten()
        for i in range(nbits):
            if patch[compareX[i]] <  patch[compareY[i]]:
                test.append(1)
            else:
                test.append(0)
        desc.append(test)
    return np.stack(locs, axis=0), np.stack(desc,axis=0)

def briefLite(im):
    '''
    Inputs          Description
    --------------------------------------------------------------------------
    im              gray image with values between 0 and 1

    OUTPUTS         Description
    --------------------------------------------------------------------------
    locs            an m x 3 vector, where the first two columns are the image coordinates
                    of keypoints and the third column is the pyramid level of the keypoints
    desc            an m x nbits matrix of stacked BRIEF descriptors.
                    m: # of valid descriptors in the image and will vary
                    n: # of bits for the BRIEF descriptor
    '''
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    compareX, compareY = makeTestPattern()
    locsDoG, gaussian_pyramid = DoGdetector(im)
    locs, desc = computeBrief(im, locsDoG, gaussian_pyramid, compareX, compareY)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    Inputs          Description
    --------------------------------------------------------------------------
    desc1&          feature descriptors for image 1 & 2 
    desc2           matrix of stacked BRIEF descriptors of size= (m1/m2 * n bits)
    
    Outputs         Description
    -------------------------------------------------------------------------- 
    matches         p x 2 matrix. where the first column are indices into desc1
                    and the second column are indices into desc2
    '''
    # initial distance metric: hamming distance
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]        # pt2的坐标需要相应向右平移im1的width
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.axis('off')
    # plt.savefig(path, bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0) # remove the white border
    plt.show()

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_R.png')
    tempalte = 'incline_L.png'
    testlist = glob.glob('../data/'+tempalte)
    for test in testlist:
        testname = os.path.split(test)[-1]
        im2 = cv2.imread('../data/'+testname)
        path = '../results/'+ testname.split('.')[0]+'.png'
        locs1, desc1 = briefLite(im1)
        locs2, desc2 = briefLite(im2)
        matches = briefMatch(desc1, desc2)
        plotMatches(im1,im2,matches,locs1,locs2,path)
    
 