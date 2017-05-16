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

#include "cnpy.h"

#include "operator.h"
#include "../error_util.h"

Operator::Operator(cudnnTensorDescriptor_t inTensorDesc, Type type) : m_deleteInTensorDesc(false), m_type(type)
{
	m_inTensorDesc.push_back(inTensorDesc);
}

Operator::Operator(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, Type type) :
		m_deleteInTensorDesc(true), m_type(type)
{
	cudnnTensorDescriptor_t inTensorDesc;
	cudnnSafeCall(cudnnCreateTensorDescriptor(&inTensorDesc));
	cudnnSafeCall(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in, ic, ih, iw));
	m_inTensorDesc.push_back(inTensorDesc);
}

Operator::Operator(std::vector<cudnnTensorDescriptor_t> &inTensorDesc, Type type) :
		m_inTensorDesc(inTensorDesc), m_deleteInTensorDesc(false), m_type(type)
{
}

Operator::~Operator()
{
	if (m_deleteInTensorDesc)
		for(int idx = 0; idx < m_inTensorDesc.size(); ++idx)
			cudnnSafeCall(cudnnDestroyTensorDescriptor(m_inTensorDesc[idx]));
}

iu::TensorGpu_32f *Operator::forward(iu::TensorGpu_32f *d_inputs, cudnnHandle_t cudnnHandle)
{
	std::vector<iu::TensorGpu_32f *> d_vecInputs = { d_inputs };
	return forward(d_vecInputs, cudnnHandle);
}

void Operator::setParams(std::string npzPath, std::string array)
{
	// load from npz array
	cnpy::NpyArray params = cnpy::npz_load(npzPath, array);

	// get destination
	auto d_params = getParams();

	// load to cpu
	iu::TensorCpu_32f *h_params;
	if (params.shape.size() == 4)
		h_params = new iu::TensorCpu_32f(reinterpret_cast<float *>(params.data), params.shape[0], params.shape[1],
				params.shape[2], params.shape[3], true);
	else if (params.shape.size() == 1)
		h_params = new iu::TensorCpu_32f(reinterpret_cast<float *>(params.data), 1, params.shape[0], 1, 1, true);

	// load to device filter memory
	if (h_params->samples() != d_params->samples() || h_params->channels() != d_params->channels()
			|| h_params->height() != d_params->height() || h_params->width() != d_params->width())
	{
		std::cout << "ERROR: Loaded filters do not match the size of the convolution filters" << std::endl;
		std::cout << "Loaded shape " << params.shape[0] << " " << params.shape[1] << " " << params.shape[2] << " " << params.shape[3] << ", but shape should be " << d_params->size() << std::endl;
		delete h_params;
		exit(-1);
	}

	iu::copy(h_params, d_params);
	//delete[] params.data;
	params.destruct();
	delete h_params;
}

void Operator::setParams(iu::TensorCpu_32f *h_params)
{
	// get destination
	auto d_params = getParams();

	// load to device filter memory
	if (h_params->samples() != d_params->samples() || h_params->channels() != d_params->channels()
			|| h_params->height() != d_params->height() || h_params->width() != d_params->width())
	{
		std::cout << "ERROR: Loaded filters do not match the size of the convolution filters" << std::endl;
		delete h_params;
		exit(-1);
	}

	iu::copy(h_params, d_params);
}
