#pragma once
#include "iu/iucore.h"

namespace cuda
{

void forward(iu::TensorGpu_32f &d_outNHWC, iu::TensorGpu_32f &d_inLeft,
		iu::TensorGpu_32f &d_inRight, iu::LinearDeviceMemory_32f_C1 &d_disparities, int rectCorr);

void forward(float *d_out, float *d_inLeft, float *d_inRight, float *d_disparities, int in, int ic,
		int ih, int iw, int numDisps, iu::TensorGpu_32f::MemoryLayout memoryLayout, int rectCorr);

void backward(float *d_outGrad0, float *d_outGrad1, float *d_inGrad, float *d_im0, float *d_im1,
		float *d_disparities, int n, int c, int h, int w, int numDisps,
		iu::TensorGpu_32f::MemoryLayout memoryLayout);

}
