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

#include <iostream>
#include <string>

#include "cnpy.h"

#include "convolution.h"
#include "../error_util.h"

Convolution::Convolution(cudnnConvolutionMode_t convMode, cudnnTensorDescriptor_t inTensorDesc,
		cudnnFilterDescriptor_t filterDesc, iu::TensorGpu_32f *d_out, int pad_x, int pad_y) :
		Operator(inTensorDesc, Operator::Type::CONVOLUTION), m_convMode(convMode), m_filterDesc(filterDesc), m_deleteFilterDesc(
				false)
{
	initialize(d_out, pad_x, pad_y);
}

Convolution::Convolution(cudnnConvolutionMode_t convMode, cudnnTensorDescriptor_t inTensorDesc, unsigned int fk,
		unsigned int fc, unsigned int fh, unsigned int fw, iu::TensorGpu_32f *d_out, int pad_x, int pad_y) :
		Operator(inTensorDesc, Operator::Type::CONVOLUTION), m_convMode(convMode), m_deleteFilterDesc(true)
{
	cudnnSafeCall(cudnnCreateFilterDescriptor(&m_filterDesc));
    cudnnSafeCall(cudnnSetFilter4dDescriptor(m_filterDesc, CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, fk, fc, fh, fw));

	initialize(d_out, pad_x, pad_y);
}

Convolution::Convolution(cudnnConvolutionMode_t convMode, unsigned int in, unsigned int ic, unsigned int ih,
		unsigned int iw, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw, iu::TensorGpu_32f *d_out, int pad_x, int pad_y) :
		Operator(in, ic, ih, iw, Operator::Type::CONVOLUTION), m_convMode(convMode), m_deleteFilterDesc(true)
{
	cudnnSafeCall(cudnnCreateFilterDescriptor(&m_filterDesc));
    cudnnSafeCall(cudnnSetFilter4dDescriptor(m_filterDesc, CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, fk, fc, fh, fw));

	initialize(d_out, pad_x, pad_y);
}

void Convolution::allocateOutputMemory(iu::TensorGpu_32f *d_out)
{
	// create output tensor
	int outN, outC, outH, outW;
	cudnnSafeCall(
			cudnnGetConvolution2dForwardOutputDim(m_convDesc, m_inTensorDesc[0], m_filterDesc, &outN, &outC, &outH,
					&outW));

	cudnnSafeCall(cudnnCreateTensorDescriptor(&m_outTensorDesc));
	cudnnSafeCall(
			cudnnSetTensor4dDescriptor(m_outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW));

	if(d_out == NULL)
	{
		m_output = new iu::TensorGpu_32f(outN, outC, outH, outW);
		m_extOutPtr = false;
	}
	else
	{
		if(d_out->samples() != outN)
		{
			std::cout << "[ConvolutionOp] Error: provided output buffer has wrong size! N = " << d_out->samples() << ", but should be " << outN << "!" << std::endl;
			exit(-1);
		}
		if(d_out->channels() != outC)
		{
			std::cout << "[ConvolutionOp] Error: provided output buffer has wrong size! C = " << d_out->channels() << ", but should be " << outC << "!" << std::endl;
			exit(-1);
		}
		if(d_out->height() != outH)
		{
			std::cout << "[ConvolutionOp] Error: provided output buffer has wrong size! H = " << d_out->height() << ", but should be " << outH << "!" << std::endl;
			exit(-1);
		}
		if(d_out->width() != outW)
		{
			std::cout << "[ConvolutionOp] Error: provided output buffer has wrong size! W = " << d_out->width() << ", but should be " << outW << "!" << std::endl;
			exit(-1);
		}
		if(d_out->memoryLayout() != iu::TensorGpu_32f::MemoryLayout::NCHW)
		{
			std::cerr << "[ConvolutionOp] Error: Provided output has wrong memory layout!" << std::endl;
			exit(-1);
		}

		m_output = d_out;
		m_extOutPtr = true;
	}
}

void Convolution::initialize(iu::TensorGpu_32f *d_out, int pad_x, int pad_y)
{
	// convolution settings
	cudnnSafeCall(cudnnCreateConvolutionDescriptor(&m_convDesc));
	cudnnSafeCall(cudnnSetConvolution2dDescriptor(m_convDesc, pad_y, pad_x, 1, 1, 1, 1, m_convMode, CUDNN_DATA_FLOAT));

	allocateOutputMemory(d_out);

	// allocate memory for filters
	int fk, fc, fh, fw;
	cudnnDataType_t dataType;
    cudnnTensorFormat_t filterFormat;
    cudnnSafeCall(cudnnGetFilter4dDescriptor(m_filterDesc, &dataType, &filterFormat, &fk, &fc, &fh, &fw));
	m_filter = new iu::TensorGpu_32f(fk, fc, fh, fw);
}

Convolution::~Convolution()
{
	cudnnSafeCall(cudnnDestroyConvolutionDescriptor(m_convDesc));
	cudnnSafeCall(cudnnDestroyTensorDescriptor(m_outTensorDesc));

	if (m_deleteFilterDesc)
		cudnnSafeCall(cudnnDestroyFilterDescriptor(m_filterDesc));

	delete m_filter;
	deleteOutputMemory();
}

void Convolution::deleteOutputMemory()
{
	if(!m_extOutPtr && m_output != NULL)
	{
		delete m_output;
		m_output = NULL;
	}
}

iu::TensorGpu_32f *Convolution::forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle)
{
	if(m_output == NULL)
		allocateOutputMemory(NULL);

	float alpha = 1.0;
	float beta = 0.0;
	cudnnSafeCall(
			cudnnConvolutionForward(cudnnHandle, &alpha, m_inTensorDesc[0], d_inputs[0]->data(), m_filterDesc, m_filter->data(), m_convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, m_outTensorDesc, m_output->data()));

	return m_output;
}

cudnnTensorDescriptor_t Convolution::outTensorDesc()
{
	return m_outTensorDesc;
}
