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
dependencies/slackprop
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
make
cd ../../
~~~
More information can be found at https://github.com/rogersce/cnpy. 

#### SlackProp
Compile SlackProp using the following commands:
~~~
cd dependencies/slackprop
mkdir build
cd build
cmake ..
make 
(sudo) make install
cd ../../
~~~

Additionally you must set the environment variable `SLACKPROP_ROOT` to point to the slackprop 
folder using

~~~
export SLACKPROP_ROOT=path/to/slackprop/
~~~

### Stereo-Net
At this point you should have compiled all the dependencies successfully. Please also double-check 
you have set the environment variables `IMAGEUTILITIES_ROOT` and `SLACKPROP_ROOT` correctly.

Compiling our stereo model:
~~~
mkdir build
cd build
cmake ../src
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
./stereo_img TODO: parameters
~~~

### Reproduce the numbers in the paper
* install opencv for python
* rectify train/test images using the provided script

~~~
cd undistort
python main.py <path/to/middlebury/>{training, test}H/
~~~

This command will warp `im1` such that corresponding pixels are located in the same row. The 
rectified images are saved as `im1_rectified.png` in the appropriate folder.

* run the provided evaluation script

~~~
cd ../eval
python run_middlebury_H.py ../data/middlebury-2014/EvaluationScripts/trainingH/ ../build blub!! ../data/parameters/middlebury/7-layer-H/ JMR_NEU
~~~

This will 

* run evaluation script to get numbers


TODO: correct Cmake to compile automatically in Release mode