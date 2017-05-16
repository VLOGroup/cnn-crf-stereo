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
