import numpy as np
from numpy.linalg import inv
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares, leastsq

from distort import undistort, approx_inv_undistort
from features import compute_sift_matches

def rectify_stereo_pair(im0, im1, K1, y_th, good_ratio, verbose=False):
    def cost_fun(theta, U0, U1, K):
        U1_ = undistort(theta, U1, K)

        residuals = (U0[1,:] - U1_[1,:]) / U0.shape[1]
        return residuals

    M, N, C = im0.shape

    # detect features and match them
    matches, y_diffs, kp0, kp1 = compute_sift_matches(im0, im1, y_th, good_ratio=good_ratio)

    # use only the best matches
    num_kp = len(matches)

    coords0 = np.ones((3,num_kp))
    coords1 = np.ones((3,num_kp))
    for idx, match in enumerate(matches[:num_kp]):
        coords0[:2,idx] = kp0[match[0].queryIdx].pt
        coords1[:2,idx] = kp1[match[0].trainIdx].pt

    # optimize for unknown distortion parameters
    if verbose:
        print 'Start optimization'
   
    theta = np.zeros(6)
    res_nlq = least_squares(cost_fun, theta, method='trf', loss='huber', ftol=1e-9, verbose=0, 
                            max_nfev=5000, args=(coords0, coords1, K1))
    
    if verbose:
        print 'Found Solution = ', res_nlq.x, 
        print 'Optimization status =', res_nlq.status

    x_opt = res_nlq.x

    # initialize distorted pixel coords
    xx, yy = np.meshgrid(np.arange(N), np.arange(M))

    # compute query coords by using the inverse distortion function
    undistorted = approx_inv_undistort(x_opt, np.vstack((xx.ravel(), 
                                                        yy.ravel(), 
                                                        np.ones_like(yy.ravel()))), K1)

    # set query coordinates (keep x fixed to not change disparities!)
    X_query = xx
    Y_query = undistorted[1].reshape((M,N))

    if verbose:
        print 'Warp Image 1 ...'
    
    im1_warped = np.concatenate((
        map_coordinates(im1[..., 0], 
                        np.vstack((Y_query.ravel(), 
                                   X_query.ravel())), mode='reflect').reshape((M,N,1)),
        map_coordinates(im1[..., 1], 
                        np.vstack((Y_query.ravel(), 
                                   X_query.ravel())), mode='reflect').reshape((M,N,1)),
        map_coordinates(im1[..., 2], 
                        np.vstack((Y_query.ravel(), 
                                   X_query.ravel())), mode='reflect').reshape((M,N,1))),
        axis=2)

    return im1_warped

def approx_inv_undistort(theta, U_u, K):
    """
    Approximiate the inverse of the distortion function by iteration (ignoring
        the dependency on the tangential parameters)
    :param theta: parameter vector [k1, p1, p2, df, dcx, dcy]
    :param U_u: undistorted pixel coordinates with shape [3 x MN]
    :param K: camera matrix
    :return: the estimated undistorted pixel coordinates
    """

    k1, p1, p2, df, dcx, dcy = theta

    # update camera mat
    K_copy = K.copy()
    K_copy[0, 0] = K[0, 0] + df
    K_copy[1, 1] = K[1, 1] + df
    K_copy[0, 2] = K[0, 2] + dcx
    K_copy[1, 2] = K[1, 2] + dcy

    # compute normalized coordinates (undistorted)
    X_u = np.dot(inv(K_copy), U_u)

    # initial guess: distorted coords are same as undistorted
    X_d = X_u.copy()

    # iterate until convergence
    for _ in np.arange(100):
        r_2 = X_d[0, :] ** 2 + X_d[1, :] ** 2

        radial_coef = (1. + k1 * r_2) #* np.array([[X_d[0, :]],
                                 #            [X_d[1, :]]]).squeeze()

        tangential = np.array([[2 * p1 * X_d[0, :] * X_d[1, :] + p2 * (r_2 + 2 * X_d[0, :] ** 2)],
                               [p1 * (r_2 + 2 * X_d[1, :] ** 2) + 2 * p2 * X_d[0, :] * X_d[1, :]]]).squeeze()

        X_d_old = X_d.copy()
        X_d[:2] = (X_u[:2] - tangential) / radial_coef

        if np.abs(X_d_old - X_d).max() < 1e-15:
            # print 'Break because of convergence at iteration ' + str(_)
            break

    # compute pixel coordinates (distorted)
    U_d = np.dot(K_copy, X_d)
    return U_d

def undistort(theta, U_d, K):
    """ Computes undistorted pixel coordinates given the camera matrix K and a parameter-vector theta
        @param theta: parameter vector [k1, p1, p2, df, dcx, dcy]
        @param U_d: distorted pixel coordinates with shape [3 x MN]
        @param K: camera matrix
    """

    k1, p1, p2, df, dcx, dcy = theta

    # update camera mat
    K_copy = K.copy()
    K_copy[0,0] = K[0,0] + df
    K_copy[1,1] = K[1,1] + df
    K_copy[0,2] = K[0,2] + dcx
    K_copy[1,2] = K[1,2] + dcy

    # compute normalized coordinates (undistorted)
    X_d = np.dot(inv(K_copy), U_d)

    # compute undistorted normalized points
    r_2 = X_d[0, :] ** 2 + X_d[1, :] ** 2

    radial = (1. + k1 * r_2) * np.array([[X_d[0, :]],
                                         [X_d[1, :]]]).squeeze()

    tangential = np.array([[2 * p1 * X_d[0, :] * X_d[1, :] + p2 * (r_2 + 2 * X_d[0, :] ** 2)],
                           [p1 * (r_2 + 2 * X_d[1, :] ** 2) + 2 * p2 * X_d[0, :] * X_d[1, :]]]).squeeze()

    X_u = radial + tangential
    X_u = np.vstack((X_u, np.ones(X_u.shape[1])))

    # compute pixel coordinates (undistorted)
    U_u = np.dot(K_copy, X_u)
    return U_u