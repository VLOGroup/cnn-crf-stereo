import sys
import glob
import os
import subprocess
import pickle
import numpy as np

if len(sys.argv) < 6:
    print 'Usage: python run_kitti_2015.py <input_folder> <executable_dir> <config_file> <cnn_params_folder> <output_folder> <num_files>'
    sys.exit(-1)

data_dir = sys.argv[1]
executable_dir = sys.argv[2]
config_file = sys.argv[3]
cnn_params_folder = sys.argv[4]
output_dir = sys.argv[5]
if len(sys.argv) == 7:
    num_images = int(sys.argv[6])
else:
    num_images = 200

image0_dir = data_dir + 'image_2/'
image1_dir = data_dir + 'image_3/'
executable = executable_dir + '/stereo_img'
config_file = cnn_params_folder + '/' + config_file

if not os.path.exists(image0_dir):
    print 'Path to image0 does not exist!'
    sys.exit(-2)

if not os.path.exists(image1_dir):
    print 'Path to image1 does not exist!'
    sys.exit(-2)

if not os.path.isfile(executable):
    print 'Stereo_img does not exist! Did you build stereo_img already?!'
    sys.exit(-2)

if not os.path.isfile(config_file):
  print 'Config file not found!'
  sys.exit(-2)

# create npz file from parameters
pkl_names = glob.glob(cnn_params_folder + '/*.pkl')
print 'Using', pkl_names[0]
with open(pkl_names[0], 'r') as pkl_file:
    params = pickle.load(pkl_file)
cnn_params_file = 'params'

if os.path.exists(cnn_params_file+'.npz'):
    os.remove(cnn_params_file+'.npz')
with open(cnn_params_file+'.npz','w') as f:
    np.savez(f, *params)

# get all image names
image0_names = sorted(glob.glob(image0_dir + '*10.png'))
image1_names = sorted(glob.glob(image1_dir + '*10.png'))

# create output dir
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    os.makedirs(output_dir)
except:
    pass
    #print 'Error: Could not create output dir', output_dir
    #sys.exit(-3)

for image0, image1, imageid in zip(image0_names, image1_names, np.arange(1,201)):
    print os.path.basename(image0)

    #CNN + CRF output
    args = [executable, image0, image1, "--config-file " + config_file,
                                     '--output-file ' + output_dir + '/' + os.path.splitext(os.path.basename(image0))[0],
                                     '--parameter-file', cnn_params_file,
                                     #'--refinement','QuadDirect'
                                     ]
    print args
    retval = subprocess.call(' '.join(args), shell=True, cwd=os.getcwd())
    if imageid==num_images:
        break
