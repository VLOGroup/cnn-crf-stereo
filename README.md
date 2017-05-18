# Hybrid CNN-CRF Stereo 
This repository provides software for our publication "End-to-End Training of Hybrid CNN-CRF Models 
for Stereo", which is going to be presented at CVPR 2017.

If you use this code please cite the following publication:
~~~
@inproceedings{knoebelreiter_cvpr2017,
  author = {Patrick Knöbelreiter and Christian Reinbacher and Alexander Shekhovtsov and Thomas Pock},
  title = {{End-to-End Training of Hybrid CNN-CRF Models for Stereo}},
  booktitle = {2017 Computer Vision and Pattern Recognition (CVPR)},
  year = {2017},
}
~~~

## Repository Structure
The repository is structured as follows:
  - `stereo` contains the C++/CUDA code of our CNN-CRF model -> this needs to be compiled to use our model for disparity computation
  - `undistort` contains python code for rectifying the Middlebury images
  - `eval` contains python code for reproducing the numbers presented in the paper
  - `data` contains the learned model parameters

## Compiling
For your convenience, the required libraries that are on Github are added as
submodules. So clone this repository with `--recursive` or do a
~~~
git submodule update --init --recursive
~~~
after cloning. All dependencies will be located in the dependency folder like
~~~
dependencies/cnpy
dependencies/imageutilities
dependencies/slack-prop
~~~

<!---If you are already using some projects from our group, 
(https://github.com/VLOGroup/), a recursive clone is not necessary in order to avoid having the 
same code twice on your machine. In either case you need to set environment variables for each 
project, such that CMake can find the dependencies. Setting the environment variables is described 
below.-->

### Dependencies
This software requies:
 - GCC >= 4.9
 - CMake >= 3.2
 - ImageUtilities (https://github.com/VLOGroup/imageutilities) with the `iuio` and `iumath` modules
 - SlackProp (https://github.com/VLOGroup/slackprop)
 - cnpy (https://github.com/rogersce/cnpy)
 - cudnn 6.x (https://developer.nvidia.com/cudnn)

#### Image Utilities
Compile and install imageutilities: Follow the instructions on 
https://github.com/VLOGroup/imageutilities. Make sure you set the environment variable 
`IMAGEUTILITIES_ROOT` correctly. This is necessary to find the compiled
library automatically with CMake.

#### cnpy
Compile and install cnpy by executing the following commands:
 ~~~
cd dependencies/cnpy
mkdir build
cd build
cmake ..
(sudo) make install
cd ../../
~~~
More information can be found at https://github.com/rogersce/cnpy. 

#### SlackProp
Compile SlackProp using the following commands:
~~~
cd dependencies/slack-prop
mkdir build
cd build
cmake ..
make 
cd ../../
~~~

<!---Additionally you must set the environment variable `SLACKPROP_ROOT` to point to the slackprop 
folder using

~~~
export SLACKPROP_ROOT=path/to/slackprop/
~~~
--->
### Stereo-Net
At this point you should have compiled all the dependencies successfully. Please also double-check 
you have set the environment variables `IMAGEUTILITIES_ROOT` and `SLACKPROP_ROOT` correctly.

Compiling our stereo model:
~~~
mkdir build
cd build
cmake ../stereo
make
~~~

If everything worked correctly, you should see an executeable called `stereo_img` in your build 
directory. The following simple test will print the usage information 
~~~
./stereo_img
~~~

## Usage
In order to demonstrate the usage of our code we put a rectified stereo-pair into the data
directory. You can compute the disparity map using
~~~
./stereo_img --im0 ../data/im0.png --im1 ../data/im1.png
~~~
This will create a file called `output.png` in the same directory.
Otherwise, download the Middlebury data as described below. Then you can 
test the algorithm using
~~~
./stereo_img --im0 ../data/middlebury-2014/MiddEval3/trainingQ/Adirondack/im0.png  --im1 ../data/middlebury-2014/MiddEval3/trainingQ/Adirondack/im1.png --parameter-file ../data/parameters/middlebury-2014/7-layer/cnn+crf+full/params --config-file ../data/parameters/middlebury-2014/7-layer/cnn+crf+full/config_mb_cnn7_crf_full.cfg
~~~

### Reproduce the numbers in the paper
#### Dependencies
* OpenCV >= 3.0 (https://github.com/opencv/opencv) with python bindings 
* OpenCV contribution package `xfeatures2d` (https://github.com/opencv/opencv_contrib) with python bindings
* scipy >= 0.17 
* The C-Shell csh

#### Datasets
First you must download the data from the respective benchmark and then you can use the provided 
evaluation scripts to reproduce the numbers in the paper.

Create the two directories `middlebury-2014` and `kitti-2015` in the `data` directory, such that the
structure of the data directory is
~~~
data/kitti-2015/
data/middlebury-2014/
~~~

#### Middlebury Stereo Evaluation - Version 3 
1. Download the Middlebury 2014 data from (http://vision.middlebury.edu/stereo/submit3/). You will 
will need `Input Data` as well as `Ground truth for left view` for quarter (Q), half (H) and full (F) resolution. 
2. Download the Middlebury evaluation SDK (http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-SDK-1.6.zip)
3. Extract all downloaded files to the `data` folder. 
Your data folder should look like 
~~~
data/
|--- middlebury-2014/
    |--- MiddEval3/
        |--- trainingQ/
        |--- trainingH/
        |--- testQ/
        |--- testH/
        |--- runevalF
        |--- ...
~~~
4. Compile the Middlebury evaluation SDK like described in http://vision.middlebury.edu/stereo/submit3/zip/MiddEval3/README.txt
5. Rectify train/test images using the provided script
~~~
cd undistort
python main.py ../data/middlebury-2014/MiddEval3/trainingH/
python main.py ../data/middlebury-2014/MiddEval3/testH/
~~~

This command will warp `im1` such that corresponding pixels are located in the same row. The 
rectified images are saved as `im1_rectified.png` in the appropriate folder.

6. Compute results for Middlebury
~~~
cd eval
./run_all_middlebury.sh
~~~

7. Compute Errors
~~~
python compute_numbers_middlebury.py
~~~

#### Kitti Stereo Evaluation 2015
1. Download the Kitti 2015 data from (http://www.cvlibs.net/download.php?file=data_scene_flow.zip). 
2. Make a folder `kitti-2015` in the `data` folder and extract all downloaded files there
Your data folder should look like
~~~
data/
|--- kitti-2015
    |--- testing/
    |--- training/
~~~

3. Compute results for Kitti
~~~
cd eval
./run_all_kitti_2015.sh
~~~

4. Compute Errors
~~~
python compute_numbers_kitti.py
~~~