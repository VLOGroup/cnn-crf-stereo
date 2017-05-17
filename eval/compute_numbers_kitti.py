from os import listdir
from os.path import join, exists
import numpy as np
from skimage.io import imread

def readdisp16(path):
    disp = imread(path)
    disp = disp / 256.0
    disp[disp == 0] = -1
    return disp

def computeError(prediction, gt, tau):
    diff = np.abs(prediction - gt)
    wrong_elems = (gt > 0) & (diff > tau[0]) & ((diff / np.abs(gt)) > tau[1])

    error = wrong_elems.sum() / np.sum(gt[gt > 0].shape).astype('float64')
    return error

# methods = ['CNN3', 'CNN3CRF', 'CNN3CRFJOINT', 'CNN3CRFFULL',
#            'CNN7', 'CNN7CRF', 'CNN7CRFJOINT', 'CNN7CRFFULL']
methods = ['CNN3', 'CNN7', 'MBCNN3_on_Kitti', 'MBCNN7_on_Kitti',
           'MBCNNCRFJ3_on_Kitti', 'MBCNNCRFJ7_on_Kitti']

error_px = 3
error_percent = 0.05

results = {}
for method in methods:
    src_dir = join('../data/out/kitti2015/', method)
    if not exists(src_dir):
        results[method] = np.array([-1, -1])
        continue

    files = listdir(src_dir)
    files.sort()

    # keep only pngs
    files = [f for f in files if '.png' in f]
    print 'Found {} files for method {}'.format(len(files), method)

    percent_bad_occ = []
    percent_bad_noc = []
    for fname in files:
        pred = readdisp16(join(src_dir, fname))
        gt_occ = readdisp16(join('../data/kitti-2015/training/disp_occ_0', fname))
        gt_noc = readdisp16(join('../data/kitti-2015/training/disp_noc_0', fname))
        
        print fname

        # compare prediction with ground-truth
        percent_bad_occ.append(computeError(pred, gt_occ, [error_px, error_percent]))
        percent_bad_noc.append(computeError(pred, gt_noc, [error_px, error_percent]))

    results[method] = np.array([np.asarray(percent_bad_occ).mean(), 
                                np.asarray(percent_bad_noc).mean()])

        
for method in methods:
    print "{0: <13} {1}".format(method, results[method])

