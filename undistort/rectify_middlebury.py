import numpy as np
from os.path import join
from skimage.io import imread, imsave
import os
import sys
from rectify import rectify_stereo_pair

def read_middlebury_calib_file(path_to_calib_file):
    # read camera matrix
    if os.path.exists(path_to_calib_file):
        with open(path_to_calib_file) as f:
            lines = f.readlines()
            for line in lines:
                if 'cam1' in line:
                    number_list = line[6:-2].replace(';', '').split(' ')

                    cam1 = np.zeros((3,3))
                    for j in np.arange(3):
                        for i in np.arange(3):
                            cam1[i,j] = number_list[j + i*3]
        
        return cam1
    else:
        raise IOError("Error: Could not find file: " + path_to_calib_file)

# main code
if len(sys.argv) != 2:
    print 'Usage: python rectify_middlebury.py <img_dir>'
    exit(-1)

img_dir = sys.argv[1]

DEBUG = False
np.set_printoptions(suppress=True, precision=5, linewidth=200)

# middlebury evaluation: training and testing
im0 = imread(join(img_dir, 'im0.png'))
im1 = imread(join(img_dir, 'im1.png'))
K1 = read_middlebury_calib_file(join(img_dir, 'calib.txt'))

# set parameters
y_th = 3
good_ratio = 0.75

# compute result
im1_rectified = rectify_stereo_pair(im0, im1, K1, y_th, good_ratio)

# write result to disk
imsave(join(img_dir, 'im1_rectified.png'), im1_rectified)