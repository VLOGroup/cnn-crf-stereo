import subprocess
from subprocess import Popen, PIPE
from os import listdir, chdir, getcwd
from os.path import join
import numpy as np
import re
np.set_printoptions(precision=2)

methods = ['CNN3', 'CNN3CRF', 'CNN3CRFJOINT', 'CNN3CRFFULL',
           'CNN7', 'CNN7CRF', 'CNN7CRFJOINT', 'CNN7CRFFULL']

# save working directory
cwd = getcwd()

# change cwd to 
chdir(join(cwd, 'data/middlebury'))

weights = np.array([1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 1, 0.5, 1, 0.5])
results = {}
for method in methods:
    #print 'Compute results for', method, '...'
    #res = subprocess.call(['./runevalF', '-b', "Q", "all", "4.0", method]
    p = Popen(['./runevalF', '-b', "Q", "all", "4.0", method], stdin=PIPE, stdout=PIPE, stderr=PIPE)
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