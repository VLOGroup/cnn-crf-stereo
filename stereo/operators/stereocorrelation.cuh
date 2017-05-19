// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Patrick Knöbelreiter <knoebelreiter at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// cnn-crf-stereo is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// cnn-crf-stereo is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
