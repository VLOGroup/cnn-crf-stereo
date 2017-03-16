#! /bin/bash
# Q
#3-layer
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn3.cfg data/parameters/middlebury/3-layer/cnn/ disp0CNN3
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn3_crf.cfg data/parameters/middlebury/3-layer/cnn/ disp0CNN3CRF
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn3_crf_joint.cfg data/parameters/middlebury/3-layer/cnn+crf/ disp0CNN3CRFJOINT
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn3_crf_full.cfg data/parameters/middlebury/3-layer/cnn+crf+full/ disp0CNN3CRFFULL

#7-layer
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn7.cfg data/parameters/middlebury/7-layer/cnn/ disp0CNN7
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn7_crf.cfg data/parameters/middlebury/7-layer/cnn/ disp0CNN7CRF
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn7_crf_joint.cfg data/parameters/middlebury/7-layer/cnn+crf/ disp0CNN7CRFJOINT
python run_middlebury.py data/middlebury/trainingQ ../build config_mb_cnn7_crf_full.cfg data/parameters/middlebury/7-layer/cnn+crf+full/ disp0CNN7CRFFULL

# H
#3-layer
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn3_crf.cfg data/parameters/middlebury/3-layer/cnn/ disp0CNN3CRF
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn3_crf_joint.cfg data/parameters/middlebury/3-layer/cnn+crf/ disp0CNN3CRFJOINT
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn3_crf_full.cfg data/parameters/middlebury/3-layer/cnn+crf+full/ disp0CNN3CRFFULL

#7-layer
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn7_crf.cfg data/parameters/middlebury/7-layer/cnn/ disp0CNN7CRF
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn7_crf_joint.cfg data/parameters/middlebury/7-layer/cnn+crf/ disp0CNN7CRFJOINT
python run_middlebury_H.py data/middlebury/trainingH ../build config_mb_cnn7_crf_full.cfg data/parameters/middlebury/7-layer/cnn+crf+full/ disp0CNN7CRFFULL
