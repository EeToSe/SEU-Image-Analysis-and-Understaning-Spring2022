
from cv2 import Sobel
import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    '''Produces a Gaussian pyramid

    Inputs:
        im - source image
        sigma0 - standard deviation of gaussian function
        k - related to sigma_
        levels - level list of the pyramid 
    '''
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))

    # join a sequence of array along a new axis, result.size=(imH, imW, #levels) 
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid
    
def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''Produces DoG Pyramid
    Inputs:
    Gaussian Pyramid - A matrix of grayscale images of size
                        (imH, imW, #levels)
    levels      - the levels of the pyramid where the blur at each level is

    Outputs:
    DoG Pyramid - size (imH, imW, #levels-1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    DoG levels - levels[1:], which specifies corresponding levels of DoG Pyramid
    '''
    DoG_levels = levels[1:]
    DoG_pyramid = []

    for i in DoG_levels:
        # compute DoG use gp_{l} - gp_{l-1}, gl aka gaussian pyramid
        DoG_pyramid.append(gaussian_pyramid[:, :, i+1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels 

def computePrincipalCurvature(DoG_pyramid):
    ''' Computes principal curvature ratio R from DoG_pyramid
    Inputs:
        DoG Pyramid - size (imH, imW, #levels-1) matrix of the DoG pyramid
    
    Outputs:
        principal_curvature - size (imH, imW, #levels-1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid.
    '''
    principal_curvature = np.zeros(DoG_pyramid.shape)

    for l in range(DoG_pyramid.shape[-1]):
        img = DoG_pyramid[:,:,l]
        # Sobel(img,ddepth,dx,dy) omit kernel size,
        dx = cv2.Sobel(img, -1, 1, 0)
        dy = cv2.Sobel(img, -1, 0, 1)

        # Hessian matrix elements dxx, dxy, dyy
        dxx = cv2.Sobel(dx, -1, 1, 0)
        dxy = cv2.Sobel(dx, -1, 0, 1)
        dyy = cv2.Sobel(dy, -1, 0, 1)
        
        # deal with zeros in det
        det = dxx*dyy - dxy**2
        det[det==0.] = 10**(-10)

        principal_curvature[:,:,l] = (dxx + dyy)**2 /det
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - lower bound for local extremums 
        th_r        - upper bound for local extremums
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together
    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].
    sigma0          Scale of the 0th image pyramid.
    k               Pyramid Factor.  Suggest sqrt(2).
    levels          Levels of pyramid to construct. Suggest -1:4.
    th_contrast     DoG contrast threshold.  Suggest 0.03.
    th_r            Principal Ratio threshold.  Suggest 12.
    Outputs         Description
    --------------------------------------------------------------------------
    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.
    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels=levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                              th_contrast=th_contrast, th_r=th_r)

    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('panorama/data/model_chickenbroth.jpg')
    
    # test gaussian pyramid
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # # test get local extrema
    th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # # test DoG detector
    # locsDoG, gaussian_pyramid = DoGdetector(im)

    # tmp_im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
    # for point in list(locsDoG):
    #     cv2.circle(tmp_im, (2*point[0], 2*point[1]), 2, (0, 255, 0), -1)
    # cv2.imwrite('./detected_keypoints.jpg', tmp_im)