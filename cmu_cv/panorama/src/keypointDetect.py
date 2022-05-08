import numpy as np
import cv2
from scipy.ndimage import maximum_filter, minimum_filter

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
        # compute kernel with EXTENT = 3*sigma
        size = int(np.floor(3*sigma_*2) + 1)
        im_pyramid.append(cv2.GaussianBlur(im, (size,size), sigma_))

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
    return im_pyramid

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
        # compute DoG use gp_{l} - gp_{l-1}, gp aka gaussian pyramid
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
    Inputs:
        DoG_pyramid - size (imH, imW, #levels-1) matrix of the DoG pyramid
        DoG_levels  - levels[1:], which specifies corresponding levels of DoG Pyramid
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                            curvature ratio R for corresponding point in DoG_pyramid
        th_contrast - lower bound for local extrema of principal_curvature
        th_r        - upper bound for local extrema of principal_curvature
     Outputs:
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
                  scale and space, and also satisfies the two thresholds.
    '''
    imH, imW, levels = DoG_pyramid.shape
    res = []
    for l in range(levels):
        # region of interest
        region = DoG_pyramid[1:imH-1, 1:imW-1, l]
        # calculate local extremas in space
        space_max = maximum_filter(region, size=3)
        space_min = minimum_filter(region, size=3)
        # calculate local extremas in scale
        scale = []
        if l > DoG_levels[0]:
            scale.append(DoG_pyramid[1:imH-1, 1:imW-1, l-1])
        if l < DoG_levels[-1]:
            scale.append(DoG_pyramid[1:imH-1, 1:imW-1, l+1])
        scale = np.asarray(scale)
        scale_max = maximum_filter(np.max(scale, axis=0), size=3)
        scale_min = minimum_filter(np.min(scale, axis=0), size=3)
        # find local extremas in both scale and space
        is_extrema = (region >= np.maximum(space_max, scale_max)) | \
                    (region <= np.minimum(space_min, scale_min))
        is_extrema &= (np.abs(region) >= th_contrast) # remove small DoG response
        is_extrema &= (np.abs(principal_curvature[1:imH-1, 1:imW-1, l]) <= th_r) # remove large PCR

        coordinates = (np.asarray(np.where(is_extrema == True))+1).T  # +1 back to original frame
        # swap columns such that (x, y)
        coordinates[:, [1,0]] = coordinates[:, [0,1]]
        level = l*np.ones((coordinates.shape[0],1),dtype=np.uint8)
        res.append(np.hstack((coordinates, level)))

    return np.concatenate(res)
    

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
    im = cv2.imread('../data/model_chickenbroth.jpg')
    
    # test gaussian pyramid
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12

    # test DoG detector
    keypoints, gaussian_pyramid = DoGdetector(im)

    tmp_im = cv2.resize(im, (2*im.shape[1], 2*im.shape[0]))
    for point in keypoints:
        cv2.circle(tmp_im, (2*point[0], 2*point[1]), 2, (0, 255, 0), -1)
    cv2.imwrite('../results/keypoints.png', tmp_im)