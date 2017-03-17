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
#include "cudnn.h"
#include "iu/iucore.h"

class Operator
{
public:
	enum Type { CONVOLUTION, BIAS, ACTIVATION, SOFTMAX, STEREO_CORRELATION, SLACKPROP };

	// one input
	Operator(cudnnTensorDescriptor_t inTensorDesc, Type type);
	Operator(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, Type type);

	// multiple inputs
	Operator(std::vector<cudnnTensorDescriptor_t> &inTensorDesc, Type type);
	virtual ~Operator();

	virtual iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle) = 0; // pure virtual function
	iu::TensorGpu_32f *forward(iu::TensorGpu_32f *d_inputs, cudnnHandle_t cudnnHandle);

	// TODO: Is this a good default implementation?
	virtual cudnnTensorDescriptor_t outTensorDesc() { return m_inTensorDesc[0]; }
	virtual iu::TensorGpu_32f *getParams() { return NULL; }

	void setParams(std::string npzPath, std::string array);
	void setParams(iu::TensorCpu_32f *h_params);

	Type type() { return m_type; }

protected:
	std::vector<cudnnTensorDescriptor_t> m_inTensorDesc;

private:
	bool m_deleteInTensorDesc;
	Type m_type;


};
