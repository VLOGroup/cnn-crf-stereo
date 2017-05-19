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

#include "cnpy.h"

#include "bias.h"
#include <iostream>

Bias::Bias(cudnnTensorDescriptor_t inTensorDesc, cudnnTensorDescriptor_t biasTensorDesc) :
	Operator(inTensorDesc, Operator::Type::BIAS),
	m_biasTensorDesc(biasTensorDesc),
	m_deleteBiasDesc(false)
{
	initialize();
}

Bias::Bias(cudnnTensorDescriptor_t inTensorDesc, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw) :
	Operator(inTensorDesc, Operator::Type::BIAS),
	m_deleteBiasDesc(true)
{
	cudnnSafeCall(cudnnCreateTensorDescriptor(&m_biasTensorDesc));
	cudnnSafeCall(cudnnSetTensor4dDescriptor(m_biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fk, fc, fh, fw));

	initialize();
}

Bias::~Bias()
{
	delete m_bias;

	if(m_deleteBiasDesc)
		cudnnSafeCall(cudnnDestroyTensorDescriptor(m_biasTensorDesc));
}

void Bias::initialize()
{
	cudnnDataType_t dataType;
	int n, c, h, w, nStride, cStride, hStride, wStride;
	cudnnSafeCall(cudnnGetTensor4dDescriptor(m_biasTensorDesc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));

	m_bias = new iu::TensorGpu_32f(n, c, h, w);
}

iu::TensorGpu_32f *Bias::forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle)
{
	float alpha = 1.0f;
	float beta = 1.0f;
	//float beta = 0.0f;

	cudnnSafeCall(cudnnAddTensor(cudnnHandle, &alpha, m_biasTensorDesc, m_bias->data(), &beta, outTensorDesc(), d_inputs[0]->data()));

	return d_inputs[0];
}

