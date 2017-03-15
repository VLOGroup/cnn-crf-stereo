# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:27:00 2016

@author: christian
"""

import sys
import glob
import os
import subprocess
import re
import numpy as np
import pickle

if len(sys.argv) != 6:
    print 'Usage: python run_middlebury.py <image_folder> <executable_dir> <config_file> <cnn_params_folder> <output_filename (no path or extension)>'
    sys.exit(-1)

data_dir = sys.argv[1]
executable_dir = sys.argv[2]
config_file = sys.argv[3]
cnn_params_folder = sys.argv[4]
output_filename = sys.argv[5]

#DEBUG
#data_dir = '/home/christian/data/testdata/stereo_testdata/eval/MiddEval3/trainingQ/'
#executable_dir = '/home/christian/workdir/postdoc/projects/mobile-vision-private/software/playground/stereo_chain/build/'
#config_file = 'config_middlebury.cfg'
#output_filename = 'disp0'

executable = executable_dir+'/stereo_img'
config_file = cnn_params_folder+'/'+config_file

if not os.path.isfile(config_file):
  print 'Config file not found!'
  sys.exit(-2)
  
# create npz file from parameters
pkl_names = glob.glob(cnn_params_folder+'/*.pkl')
print 'Using', pkl_names[0]
with open(pkl_names[0],'r') as pkl_file:
    params = pickle.load(pkl_file)
cnn_params_file = 'params_mb'
try:
  os.remove(cnn_params_file+'.npz')
except:
  pass
with open(cnn_params_file+'.npz','w') as f:
    np.savez(f, *params)

dir_path = os.path.dirname(os.path.realpath(__file__))
images_names = glob.glob(data_dir+'/*')
ndisp_regex = re.compile('ndisp=')

for image in images_names:
    print os.path.basename(image)
    left = image+'/im0.png'
    right = image+'/im1.png'
    # read out ndisp from calib.txt
    calibfile = open(os.path.realpath(image) + '/calib.txt','r')
    for line in calibfile:
      m=ndisp_regex.match(line)
      if m:
        max_disp = int(line[m.end():])
    calibfile.close()

#        max_disp = np.ceil(tt.max())
#            max_disp = 128
    bits = np.minimum(8.0,np.ceil(np.log2(max_disp)))
    num_disps = np.power(2.0,bits)
    #max_disp = np.minimum(max_disp,num_disps)
    print 'Num. Disparities: ' + str(max_disp) + ' in ' + str(num_disps) + ' steps'
    if max_disp / num_disps > 1:
        disp_step = 2
        #disp_step = max_disp / num_disps
    else:
        disp_step = 1#max_disp / num_disps

    #CNN + CRF output
    timed_executable = '/usr/bin/time -f ''%e'' -o '+image+'/timeJMR.txt '+executable
    args = [timed_executable,left,right,
            '--config-file '+config_file,
            '--output-file '+os.path.realpath(image)+'/'+output_filename,
            '--disp-step '+str(disp_step),
            '--disp-max ' + str(max_disp),
            '--parameter-file', cnn_params_file
            '--refinement','QuadDirect'
#            '--crf-iterations','15'
#            '--L2 ' + str(8.0/disp_step),
#            '--delta ' + str(round(max(2.0,2.0/disp_step)))
            #'--lambda ' + str(0.5 * 128.0 / num_disps *disp_step)]
            #'--lambda ' + str(0.5 * 128.0 / num_disps * max_disp / num_disps)]
            #'--lambda ' + str(0.5 * disp_step)
            ]

    print 'used args: ', args

    retval = 1
    while retval != 0:     # Just retry in case something happens
        retval = subprocess.call(' '.join(args), shell=True, cwd=os.getcwd())
