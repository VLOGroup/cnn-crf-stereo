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

#include "graystereonet.h"

#include "utils.cuh"
#include "utils.h"

GrayStereoNet::GrayStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw) :
    StereoNet(numLayers,in, ic, ih, iw)
{
}

GrayStereoNet::~GrayStereoNet()
{
}

iu::TensorGpu_32f *GrayStereoNet::predict(iu::ImageGpu_32f_C1 *d_left, iu::ImageGpu_32f_C1 *d_right)
{
	// adjust inputs (contiguous memory)
	int in = 1;
	int ic = 1;

	iu::TensorGpu_32f d_inputLeft(d_left->data(), in, ic, d_left->height(), d_left->width());
    iu::copy(d_left,&d_inputLeft);

	iu::TensorGpu_32f d_inputRight(d_right->data(), in, ic, d_right->height(), d_right->width());
	iu::copy(d_right, &d_inputRight);

	iu::IuCudaTimer cut;
	if (m_verbose)
		cut.start();

	// zero-mean, unit-variance
	cuda::makeZeroMean(d_inputLeft);
	cuda::makeUnitStd(d_inputLeft);

	cuda::makeZeroMean(d_inputRight);
	cuda::makeUnitStd(d_inputRight);

	if (m_verbose)
		std::cout << "Elapsed time GRAY (zero mean, unit variance): " << cut.elapsed() << std::endl;

	return performPrediction(&d_inputLeft, &d_inputRight);
}

iu::TensorGpu_32f *GrayStereoNet::predict(iu::ImageCpu_32f_C1 *left, iu::ImageCpu_32f_C1 *right)
{
	iu::ImageGpu_32f_C1 d_left(left->width(), left->height());
	iu::copy(left, &d_left);

	iu::ImageGpu_32f_C1 d_right(right->width(), right->height());
	iu::copy(right, &d_right);

	return predict(&d_left, &d_right);

}
