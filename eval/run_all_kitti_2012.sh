#! /bin/bash

# Run all KITTI 2012 tests in series
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn3.cfg data/parameters/kitti-2012/3-layer/cnn/ data/out/kitti2012/CNN3 $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn3_crf.cfg data/parameters/kitti-2012/3-layer/cnn/ data/out/kitti2012/CNN3CRF $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn3_crf_joint.cfg data/parameters/kitti-2012/3-layer/cnn+crf/ data/out/kitti2012/CNN3CRFJOINT $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn3_crf_full.cfg data/parameters/kitti-2012/3-layer/cnn+crf+full/ data/out/kitti2012/CNN3CRFFULL $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn7.cfg data/parameters/kitti-2012/7-layer/cnn/ data/out/kitti2012/CNN7 $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn7_crf.cfg data/parameters/kitti-2012/7-layer/cnn/ data/out/kitti2012/CNN7CRF $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn7_crf_joint.cfg data/parameters/kitti-2012/7-layer/cnn+crf/ data/out/kitti2012/CNN7CRFJOINT $1
python run_kitti_2012.py data/kitti/2012/training/ ../build/ config_kitti_2012_cnn7_crf_full.cfg data/parameters/kitti-2012/7-layer/cnn+crf+full/ data/out/kitti2012/CNN7CRFFULL $1

