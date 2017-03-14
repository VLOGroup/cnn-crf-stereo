#pragma once
#include "stereonet.h"

class GrayStereoNet : public StereoNet
{
  public:
	GrayStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw);
	~GrayStereoNet();

	iu::TensorGpu_32f *predict(iu::ImageGpu_32f_C1 *d_left, iu::ImageGpu_32f_C1 *d_right);
	iu::TensorGpu_32f *predict(iu::ImageCpu_32f_C1 *left, iu::ImageCpu_32f_C1 *right);

};
