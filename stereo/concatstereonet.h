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
#include "colorstereonet.h"

class ConcatStereoNet : public ColorStereoNet
{
  public:
	ConcatStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw);
	~ConcatStereoNet();


//private:
	virtual void initNet(float min_disp=0, float max_disp=0, float step=0, int rect_corr=0,
						iu::TensorGpu_32f *d_unaryOut=NULL,	iu::TensorGpu_32f *d_pairwiseOut=NULL);

	iu::TensorGpu_32f	*m_d_outLeft;
	iu::TensorGpu_32f	*m_d_outRight;

	std::vector<iu::TensorGpu_32f*> m_d_leftLayerOutputs;
	std::vector<iu::TensorGpu_32f*> m_d_rightLayerOutputs;

	int m_growthRate;
};
