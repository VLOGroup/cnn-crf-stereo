import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def compute_sift_matches(im0g, im1g, y_th=3, good_ratio=0.75, verbose=False):
    """
    Compute the SIFT matches given two images
    :param im0: first image
    :param im1: second image
    :param y_th: allowed distance in y-direction of the matches
    :param good_ratio: used to filter out low-response keypoints
    :return: the sorted good matches (based on response)
    """

    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=7)

    kp0, des0 = sift.detectAndCompute(im0g, None)
    kp1, des1 = sift.detectAndCompute(im1g, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des0, des1, k=2)

    # Apply ratio test
    good = []
    y_diffs = []
    for m, n in matches:
        if m.distance < good_ratio * n.distance:
            y_diff = kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]
            if np.abs(y_diff) < y_th:
                y_diffs.append(y_diff)
                good.append([m])

    sorted_good = sorted(good, key=lambda x: kp0[x[0].queryIdx].response, reverse=False)

    if verbose:
        plt.figure(15)
        im3 = cv2.drawKeypoints(im0g, kp0, im0g, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(im3)
        plt.title('im0 keypoints')
        plt.pause(0.1)

        plt.figure(17)
        im4 = cv2.drawKeypoints(im1g, kp1, im1g, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(im4)
        plt.title('im1 keypoints')
        plt.pause(0.1)

        im5 = cv2.drawMatchesKnn(im0g, kp0, im1g, kp1, sorted_good, im4, flags=2)

        plt.figure(16)
        plt.imshow(im5)
        plt.title('Found ' + str(len(sorted_good)) )
        plt.pause(0.1)

    return sorted_good, y_diffs, kp0, kp1