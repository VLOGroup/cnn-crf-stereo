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
#include <vector>

#include "operator.h"

class StereoCorrelation: public Operator
{
public:
    StereoCorrelation(std::vector<cudnnTensorDescriptor_t> &inTensorDesc, iu::TensorGpu_32f *d_out=NULL,float min_disp=0, float max_disp=0, float step=0, int rect_corr=0);
	~StereoCorrelation();

	// managed methods
	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs,
			cudnnHandle_t cudnnHandle);

	cudnnTensorDescriptor_t outTensorDesc();

	void setDisparities(float min_disp, float max_disp, float step, iu::TensorGpu_32f *d_out=NULL);
    void setRectificationCorrection(int value) {m_rectCorr = value;}

	static void forward(float *input0, float *input1, float *disparities, float *output, int in,
			int ic, int ih, int iw, int lenDisps, iu::TensorGpu_32f::MemoryLayout memoryLayout,
			int rectCorr);

	static void backward(float *d_outGrad0, float *d_outGrad1, float *d_inGrad, float *d_im0,
			float *d_im1, float *d_disparities, int n, int c, int h, int w, int numDisps,
			iu::TensorGpu_32f::MemoryLayout memoryLayout);

private:
	// no copies!
	StereoCorrelation(StereoCorrelation const&);
	void operator=(StereoCorrelation const&);

	cudnnTensorDescriptor_t m_outTensorDesc;
    int m_rectCorr;

	// only used if managed mode is used
	bool m_extOutPtr;
	iu::TensorGpu_32f *m_output;
	iu::LinearDeviceMemory_32f_C1 *m_disparities;

};
