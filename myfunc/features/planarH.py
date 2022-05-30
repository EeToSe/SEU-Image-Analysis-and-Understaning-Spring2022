from random import random
import numpy as np
import cv2
from myfunc.features.BRIEF import briefLite, briefMatch, plotMatches
import random

def computeH(pairs_d, pairs_s, pairsNum):
    '''
    Inputs          Description
    ---------------------------------------------------------
    pairs_d         coordinates of pairs from destination 
    pairs_s         and source images
    pairsNum        # of pairs required for the linear equation

    Outputs         Description
    ---------------------------------------------------------
    H_stod           a 3 x 3 matrix encoding the homography that 
                    best matches the linear equation Xd = H*Xs
    '''
    
    A = []
    for i in range(pairsNum):
        xd, yd = pairs_d[i,0], pairs_d[i,1]
        xs, ys = pairs_s[i,0], pairs_s[i,1]
        A.append([xs, ys, 1, 0, 0, 0, -xd*xs, -xd*ys, -xd])
        A.append([0, 0, 0, xs, ys, 1, -yd*xs, -yd*ys, -yd])
    
    u, s, vh = np.linalg.svd(np.array(A))
    H_stod = vh[-1, :].reshape((3,3))
    return H_stod/H_stod[-1,-1]

def ransacH(matches, locs1, locs2, pairsNum=4, num_iter=500, tol=4):
    '''Returns the best homography by randomly iterationg the matches with RANSAC
    Inputs              Description
    ---------------------------------------------------------
    locs1 and locs2     matrices specifying point locations in each of the images
    matches             matrix specifying matches between these two sets of point locations
    nIter               number of iterations to run RANSAC
    tol                 tolerance value for considering a point to be an inlier

    Outputs             Description
    ---------------------------------------------------------
    bestH           homography matrix with the most inliers found during RANSAC
    '''
    matchesNum = matches.shape[0]
    pt1 = locs1[matches[:,0], 0:2]
    pt2 = locs2[matches[:,1], 0:2]

    max_count = 0
    for _ in range(num_iter):
        try:
            # randomly select four point pairs and calculate the homography
            index = random.sample(range(0, matchesNum), pairsNum)
            pt1_rand = pt1[index]
            pt2_rand = pt2[index]

            # compute the homography and projective points in theory
            H2to1 = computeH(pt1_rand, pt2_rand, pairsNum)
            pt2_augment = np.c_[pt2, np.ones(pt2.shape[0])]
            pt2_project = np.matmul(H2to1, pt2_augment.T)
            pt2_project /= pt2_project[2, :]

            # compare residuals with tolerance to get inliners
            residual = pt2_project.T[:,:2] - pt1 
            dist = np.sum(residual**2, axis=1) ** 0.5
            inlier_count = dist[dist <= tol].size
            if inlier_count > max_count:
                max_count = inlier_count
                bestH = H2to1        
        except:
            print('Bad selection of 4 points pair, can not calculate homography, skip.')
    print('Matches: {}; Maximum inliners counts: {}'.format(matchesNum, max_count))
    np.save('../results/bestH.npy', bestH)
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)

    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)

    bestH = ransacH(matches, locs1, locs2, num_iter=1000, tol=2)
    print(bestH)
