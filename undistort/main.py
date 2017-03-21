from os import listdir
from os.path import join
import subprocess as sb
import sys

if len(sys.argv) != 2:
    print 'Usage: python main.py <path/to/middleburydata/'
    exit(-1)

base_dir = sys.argv[1]

img_dirs = listdir(base_dir)
img_dirs.sort()

for img_dir in img_dirs:
    print 'Rectfiy', img_dir
    sb.check_output(['python', 'rectify_middlebury.py', join(base_dir, img_dir)])
