#pragma once
#include "stereonet.h"

class ColorStereoNet : public StereoNet
{
  public:
	ColorStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw);
	virtual ~ColorStereoNet();

	iu::TensorGpu_32f *predict(iu::ImageGpu_32f_C4 *d_left, iu::ImageGpu_32f_C4 *d_right);
	iu::TensorGpu_32f *predict(iu::ImageCpu_32f_C4 *left, iu::ImageCpu_32f_C4 *right);

	iu::TensorGpu_32f *predictXY(iu::ImageGpu_32f_C4 *d_left, iu::ImageGpu_32f_C4 *d_right);
	iu::TensorGpu_32f *predictXY(iu::ImageCpu_32f_C4 *left, iu::ImageCpu_32f_C4 *right);
};
