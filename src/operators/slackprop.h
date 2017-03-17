// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Patrick Knöbelreiter <knoebelreiter at icg dot tugraz dot at>
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
#include <vector>

#include "operator.h"

class SlackProp: public Operator
{
public:
	SlackProp(cudnnTensorDescriptor_t inTensorDesc);
	~SlackProp();

	// managed methods
	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs,
			cudnnHandle_t cudnnHandle);

	cudnnTensorDescriptor_t outTensorDesc();

	void setLambda(float value);
	void setP1(float value);
	void setP2(float value);

	// attention! this will only work with tensor layout NHWC
//	static float forward(float *costVolume, float *wx, float *wy, int *output, int in, int ic,
//			int ih, int iw, float P1, float P2);

	// output is float again
	static float forward(iu::TensorGpu_32f * costVolume , iu::TensorGpu_32f * pwVolume, iu::TensorGpu_32f *output, int max_iter=5);

	// old interface, only wx and wy
	static float forward(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
				int ih, int iw, float P1, float P2, int max_iter=5);

	static float forwardVolumeOut(float *costVolume, float *wx, float *wy, float *output, int in, int ic,
			int ih, int iw, float P1, float P2, int max_iter, int delta);

//	static void backward(float *d_outGrad0, float *d_outGrad1, float *d_inGrad, float *d_im0,
//			float *d_im1, float *d_disparities, int n, int c, int h, int w, int numDisps,
//			iu::TensorGpu_32f::MemoryLayout memoryLayout);

private:
	// no copies!
	SlackProp(SlackProp const&);
	SlackProp& operator=(SlackProp const&);

	float m_P1;
	float m_P2;
	float m_lamda;

	float m_lowerBound;

	cudnnTensorDescriptor_t m_outTensorDesc;
	iu::TensorGpu_32f *m_output;

};
