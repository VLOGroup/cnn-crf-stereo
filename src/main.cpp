#include <iostream>

#include "cuda.h"
#include "cnpy.h"
#include "iu/iuio.h"
#include "iu/iucore.h"
#include "iu/iumath.h"

#include "error_util.h"
#include "graystereonet.h"
#include "colorstereonet.h"
#include "concatstereonet.h"

#include "utils.h"

#include "math.h"

int main(int argc, char **argv)
{
	if(argc < 5)
	{
		std::cout << "Usage: stereonet <numLayers> <params.npy> <left> <right> <numDisps>" << std::endl;
		exit(-1);
	}

	int num_layers = std::atoi(argv[1]);
	std::string params_path = argv[2];
	std::string left_path = argv[3];
	std::string right_path = argv[4];
	int num_disps = std::atoi(argv[5]);



	//	int useDevice = 0;
//
	int numDevices = 0;
	cudaSafeCall(cudaGetDeviceCount(&numDevices));
	std::cout << "found " << numDevices << " devices" << std::endl;

	for (int device = 0; device < numDevices; ++device)
	{
		cudaDeviceProp deviceProperties;
		cudaSafeCall(cudaGetDeviceProperties(&deviceProperties, device));
		std::cout << "device number=" << device << " info= " << deviceProperties.name << std::endl;
	}
//
	if (numDevices > 1)
		cudaSafeCall(cudaSetDevice(1));

	//exit(1);
	cudaSafeCall(cudaSetDevice(0));

//	int currentDevice;
//	cudaSafeCall(cudaGetDevice(&currentDevice));
//	std::cout << "Use device number " << currentDevice << std::endl;
//
//	std::string base_dir = "/home/patrick/PhD/Development/stereo-learning/data/AIT/archives/road_doris/";
//	std::string left_path = base_dir + "scene0.png";
//	std::string right_path = base_dir + "scene1.png";
//
//	std::string base_dir = "/home/patrick/PhD/Development/Datasets/raw_data/MB_Eval/trainingQ/Adirondack/";
//	std::string left_path = base_dir + "im0.png";
//	std::string right_path = base_dir + "im1.png";

	iu::ImageCpu_32f_C4 *im0_rgb = iu::imread_32f_C4(left_path);
	iu::ImageCpu_32f_C4 *im1_rgb = iu::imread_32f_C4(right_path);

	iu::ImageCpu_32f_C1 *left_img = iu::imread_32f_C1(left_path);
	iu::ImageCpu_32f_C1 *right_img = iu::imread_32f_C1(right_path);


//	num_layers = 3;
//	GrayStereoNet stereoNet(num_layers, 1, 1, left_img->height(), left_img->width());
//	params_path = "/home/patrick/sshfs/repos/stereo-learning/output/new/MB_3_CE/params/params3.npz";
//	int crop = 4;

//	num_layers = 3;
//	ColorStereoNet stereoNet(num_layers, 1, 3, im0_rgb->height(), im0_rgb->width());
	//params_path = "/home/patrick/sshfs_christian/stereo-learning/output/01_train_unaries/Middlebury/3_zm_std_new_001/cont/params/params3.npz";
	//params_path = "/home/patrick/PhD/development/git/stereo-learning/output/03_train_pairwise_VGG_structured/MB3_vgg_init/params/params3.npz";
	//int crop = 2*2;

	num_layers = 7;
	//ColorStereoNet stereoNet(num_layers, 1, 3, im0_rgb->height(), im0_rgb->width());
	ColorStereoNet stereoNet(num_layers, 1, 5, im0_rgb->height(), im0_rgb->width());
	//params_path = "/home/patrick/PhD/development/git/mobile-vision-private/papers/CVPR17/models/middlebury/7-layer/cnn+crf+full/params7.npz";
//#params_path = "/home/patrick/sshfs/repos/stereo-learning/output/MB/01_train_unaries/7_Color_ms/cont/params/params7.npz";
	params_path = "/home/patrick/sshfs_gottfried/stereo-learning/output/kitti-2015/04_full_joint/7-layer-xy_with_occ/params/params7.npz";
	int crop = 2*4;

//    ConcatStereoNet stereoNet(num_layers, 1, 3, im0_rgb->height(), im0_rgb->width());
//    int crop = 0;

	iu::TensorGpu_32f d_unaryOut(1, 128, left_img->height() - crop, left_img->width() - crop, iu::TensorGpu_32f::NHWC);
	iu::math::fill(d_unaryOut, 0.0f);

	iu::TensorGpu_32f d_pairwiseOut(1, 2, left_img->height() - crop, left_img->width() - crop, iu::TensorGpu_32f::NCHW);
	iu::math::fill(d_pairwiseOut, 0.0f);

    stereoNet.initNet(0.0, 127.0, 1.0, 0, &d_unaryOut, &d_pairwiseOut);
	stereoNet.setAllParams(params_path);

	float max_disp = std::pow(2.0, std::ceil(std::log2(static_cast<float>(num_disps)+1.0f))) - 1;
	stereoNet.setDisparities(0.0, max_disp, 1.0, &d_unaryOut);
	stereoNet.setVerbose(true);
	stereoNet.setAllowGc(true);


//	iu::TensorGpu_32f *d_out = stereoNet.predict(left_img, right_img);
	//iu::TensorGpu_32f *d_out = stereoNet.predict(im0_rgb, im1_rgb);
	ColorStereoNet *colorNet = dynamic_cast<ColorStereoNet*>(&stereoNet);
	auto d_out = colorNet->predictXY(im0_rgb, im1_rgb);


	// save output
	save(d_unaryOut, "/tmp/corr.npy");
	save(*d_out, "/tmp/corr_out.npy");
	save(d_pairwiseOut, "/tmp/pw.npy");



	delete left_img;
	delete right_img;
	//delete d_out;

	return 0;
}
