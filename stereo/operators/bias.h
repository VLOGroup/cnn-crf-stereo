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

#include "cudnn.h"
#include "operator.h"
#include "../error_util.h"

class Bias : public Operator
{
public:
	Bias(cudnnTensorDescriptor_t inTensorDesc, cudnnTensorDescriptor_t biasTensorDesc);
	//Bias(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, cudnnTensorDescriptor_t biasTensorDesc);
	Bias(cudnnTensorDescriptor_t inTensorDesc, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw);
	//Bias(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw);
	~Bias();

	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle);

	iu::TensorGpu_32f *getParams() { return m_bias; }

private:
	// no copies!
	Bias(Bias const&);
	void operator=(Bias const&);

	void initialize();

	iu::TensorGpu_32f *m_bias;
	cudnnTensorDescriptor_t m_biasTensorDesc;
	bool m_deleteBiasDesc;
};
