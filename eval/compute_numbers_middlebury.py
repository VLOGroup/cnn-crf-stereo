import subprocess
from subprocess import Popen, PIPE
from os import listdir, chdir, getcwd
from os.path import join
import numpy as np
import re
import sys
np.set_printoptions(precision=2)

resolution = 'Q'
if len(sys.argv) == 2:
    resolution = sys.argv[1]

if resolution == 'Q':
    methods = ['CNN3', 'CNN3CRF', 'CNN3CRFJOINT', 'CNN3CRFFULL',
               'CNN7', 'CNN7CRF', 'CNN7CRFJOINT', 'CNN7CRFFULL']
elif resolution == 'H':
    methods = ['CNN3', 'CNN3CRF', 'CNN3CRFJOINT', 'CNN3CRFFULL',
               'CNN7', 'CNN7CRF', 'CNN7CRFJOINT', 'CNN7CRFFULL',
               'JMR']
else:
    print 'Error: only resolutions Q and H are supported'
    sys.exit(-1)


# save working directory
cwd = getcwd()

# change cwd to 
chdir(join(cwd, '../data/middlebury-2014/MiddEval3'))

weights = np.array([1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 1, 0.5, 1, 0.5])
results = {}
for method in methods:
    print 'Compute results for', method, '...'
    #p = Popen(['./runevalF', method], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if resolution == 'Q':
        error_th = '4.0'
    elif resolution == 'H':
        error_th = '2.0'
    print resolution, error_th
    p = Popen(['./runevalF', '-b', resolution, 'training', error_th, method], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    return_code = p.returncode
    
    # parse output
    lines = output.splitlines()
    badx = []
    for l in lines:
        tokens = re.split(r"\s+", l)
        try:
            badx.append(np.float(tokens[3]))
        except ValueError:
            pass 
    
    if len(badx) == 0:
        results[method] = np.array([-1, -1])
        print "Warning: Could not find {} for size H".format(method)
        continue

    results[method] = np.array([np.asarray(badx).mean(), 
                                (weights * badx).sum() / weights.sum()])

for method in methods:
    print "{0: <13} {1}".format(method, results[method][1])