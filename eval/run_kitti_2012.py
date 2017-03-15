# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:27:00 2016

@author: christian
"""

import sys
import glob
import os
import subprocess
import numpy as np
import pickle

if len(sys.argv) < 6:
    print 'Usage: python run_kitti.py <input_folder> <executable_dir> <config_file> <cnn_params_folder> <output_folder> <num_files>'
    sys.exit(-1)

data_dir = sys.argv[1]
executable_dir = sys.argv[2]
config_file = sys.argv[3]
cnn_params_folder = sys.argv[4]
output_dir = sys.argv[5]
if len(sys.argv)==7:
    num_images = int(sys.argv[6])
else:
    num_images = 194

#DEBUG
#data_dir = '/home/christian/data/testdata/stereo_testdata/kitti/data_scene_flow/testing/'
#executable_dir = '/home/christian/workdir/postdoc/projects/mobile-vision-private/software/playground/stereo_chain/build/'
#config_file = 'config_kitti.cfg'
#output_dir = './out_kitti'

image0_dir = data_dir + 'image_0/'
image1_dir = data_dir + 'image_1/'
executable = executable_dir+'/stereo_img'
config_file = cnn_params_folder+'/'+config_file

# create npz file from parameters
pkl_names = glob.glob(cnn_params_folder+'/*.pkl')
print 'Using', pkl_names[0]
with open(pkl_names[0],'r') as pkl_file:
    params = pickle.load(pkl_file)
cnn_params_file = 'params'
os.remove(cnn_params_file+'.npz')
with open(cnn_params_file+'.npz','w') as f:
    np.savez(f, *params)

image0_names = sorted(glob.glob(image0_dir+'*10.png'))
image1_names = sorted(glob.glob(image1_dir+'*10.png'))
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    os.makedirs(output_dir)
except:
    pass

for image0, image1, imageid in zip(image0_names,image1_names,np.arange(1,195)):
    print os.path.basename(image0)

    #CNN + CRF output
    args = [executable,image0,image1,'--config-file',config_file,
                                     '--output-file',output_dir+'/'+os.path.splitext(os.path.basename(image0))[0],
                                     '--parameter-file', cnn_params_file,
                                     '--refinement','QuadDirect']
    retval = subprocess.call(' '.join(args),shell=True,cwd=os.getcwd())
    if imageid==num_images:
        break