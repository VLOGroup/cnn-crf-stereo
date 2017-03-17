#! /bin/bash

# Run all KITTI 2015 tests in series

python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn3.cfg ../data/parameters/kitti-2015/3-layer/cnn/ ../data/out/kitti2015/CNN3 $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn3_crf.cfg ../data/parameters/kitti-2015/3-layer/cnn/ ../data/out/kitti2015/CNN3CRF $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn3_crf_joint.cfg ../data/parameters/kitti-2015/3-layer/cnn+crf/ ../data/out/kitti2015/CNN3CRFJOINT $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn3_crf_full.cfg ../data/parameters/kitti-2015/3-layer/cnn+crf+full/ ../data/out/kitti2015/CNN3CRFFULL $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn7.cfg ../data/parameters/kitti-2015/7-layer/cnn/ ../data/out/kitti2015/CNN7 $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn7_crf.cfg ../data/parameters/kitti-2015/7-layer/cnn/ ../data/out/kitti2015/CNN7CRF $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/training/ ../build config_kitti_2015_cnn7_crf_joint.cfg ../data/parameters/kitti-2015/7-layer/cnn+crf/ ../data/out/kitti2015/CNN7CRFJOINT $1
python run_kitti_2015.py ../data/kitti/../data_scene_flow/testing/ ../build config_kitti_2015_cnn7_crf_full.cfg ../data/parameters/kitti-2015/7-layer/cnn+crf+full/ ../data/out/kitti2015/CNN7CRFJOINT $1
