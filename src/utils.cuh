// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// dvs-panotracking is free software: you can redistribute it and/or modify it under the
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
#include "cuda.h"
#include "iu/iucore.h"

namespace cuda
{
// image
float sum(iu::ImageGpu_32f_C1 &d_img);

float mean(iu::ImageGpu_32f_C1 &d_img);

float std(iu::ImageGpu_32f_C1 &d_img, float mean=-11111111);

float makeZeroMean(iu::ImageGpu_32f_C1 &d_inOut);

float makeUnitStd(iu::ImageGpu_32f_C1 &d_inOut, float mean=-11111111);

// tensor
float sum(iu::TensorGpu_32f &d_img);

float mean(iu::TensorGpu_32f &d_img);

float std(iu::TensorGpu_32f &d_img, float mean=-11111111);

void makeRgbZeroMean(iu::TensorGpu_32f &d_inOut, float *out_r_mean=NULL, float *out_g_mean=NULL, float *out_b_mean=NULL);

float makeZeroMean(iu::TensorGpu_32f &d_inOut);

void makeRgbUnitStd(iu::TensorGpu_32f &d_inOut, bool isZeroMean=false, float *out_r_std=NULL, float *out_g_std=NULL, float *out_b_std=NULL);

float makeUnitStd(iu::TensorGpu_32f &d_inOut, float mean=-11111111);

void negateTensor(iu::TensorGpu_32f &d_tensor);

void gradX(iu::TensorGpu_32f &d_inputImg, iu::TensorGpu_32f &d_gradX);

void gradY(iu::TensorGpu_32f &d_inputImg, iu::TensorGpu_32f &d_gradY);

void expTensor(iu::TensorGpu_32f &d_tensor, float alpha=1.0, float beta=1.0, float lambda=1.0);

void padColorImage(iu::ImageGpu_32f_C4 *out,iu::ImageGpu_32f_C4 *in, int padval);

void cropTensor(iu::TensorGpu_32f &d_in, iu::TensorGpu_32f &d_out, int crop);

inline uint divUp(uint a, uint b) { return (a + b - 1) / b; }

void concatOver_c_dim(iu::TensorGpu_32f &first, iu::TensorGpu_32f &second, iu::TensorGpu_32f &output);

}
