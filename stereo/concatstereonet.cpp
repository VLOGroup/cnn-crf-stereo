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

#include "concatstereonet.h"

#include "operators/convolution.h"
#include "operators/bias.h"
#include "operators/activation.h"
#include "operators/stereocorrelation.h"
#include "operators/softmax.h"

#include "utils.h"

ConcatStereoNet::ConcatStereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw) :
	ColorStereoNet(numLayers, in , ic, ih, iw), m_d_outLeft(NULL), m_d_outRight(NULL)
{
	m_growthRate = 0;
	if(numLayers == 3)
		m_growthRate = 33;
	else if(numLayers == 5)
		m_growthRate = 20;
	else if(numLayers == 7)
		m_growthRate = 14;
	else if(numLayers == 10)
		m_growthRate = 10;
	else if(numLayers == 15)
		m_growthRate = 7;
	else if(numLayers == 20)
		m_growthRate = 5;
	else if(numLayers == 25)
		m_growthRate = 4;
	else if(numLayers == 33)
		m_growthRate =  3;
	else if(numLayers == 50)
		m_growthRate = 2;
	else if(numLayers == 100)
		m_growthRate = 1;
	else
	{
		std::cout << "Error: ConcatStereoNet must have one of the following numbers of layers: 3, 5, 7, 10, 15, 20, 25, 33, 50, 100" << std::endl;
		exit(-1);
	}
}

ConcatStereoNet::~ConcatStereoNet()
{
	for(auto tensor : m_d_leftLayerOutputs)
		delete tensor;

	for(auto tensor : m_d_rightLayerOutputs)
		delete tensor;

	delete m_d_outLeft;
	delete m_d_outRight;
}


void ConcatStereoNet::initNet(float min_disp, float max_disp, float step, int rect_corr,
							iu::TensorGpu_32f *d_unaryOut,	iu::TensorGpu_32f *d_pairwiseOut)
{
	if(m_d_outLeft)
		delete m_d_outLeft;
	if(m_d_outRight)
		delete m_d_outRight;

	m_d_outLeft = new iu::TensorGpu_32f(m_in, m_growthRate * m_numLayers, m_ih, m_iw, iu::TensorGpu_32f::NCHW);
	m_d_outRight = new iu::TensorGpu_32f(m_in, m_growthRate * m_numLayers, m_ih, m_iw, iu::TensorGpu_32f::NCHW);

	// make a view of the tensor for each layer
	m_d_leftLayerOutputs.push_back(new iu::TensorGpu_32f(m_d_outLeft->data(), m_in, m_growthRate, m_ih, m_iw, true, iu::TensorGpu_32f::NCHW));
	m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_ic, m_ih, m_iw, m_growthRate, m_ic, 3, 3, m_d_leftLayerOutputs.back(), 1, 1));
	m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, m_growthRate, 1, 1));
	m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_d_rightLayerOutputs.push_back(new iu::TensorGpu_32f(m_d_outRight->data(), m_in, m_growthRate, m_ih, m_iw, true, iu::TensorGpu_32f::NCHW));
	m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_ic, m_ih, m_iw, m_growthRate, m_ic, 3, 3, m_d_rightLayerOutputs.back(), 1, 1));
	m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, m_growthRate, 1, 1));
	m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));


	for(int layerIdx = 1; layerIdx < m_numLayers; ++layerIdx)
	{
		m_d_leftLayerOutputs.push_back(new iu::TensorGpu_32f(m_d_outLeft->data(m_in*m_ih*m_iw * layerIdx*m_growthRate), m_in, m_growthRate, m_ih, m_iw, true, iu::TensorGpu_32f::NCHW));
		m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_growthRate, m_ih, m_iw, m_growthRate, m_growthRate, 3, 3, m_d_leftLayerOutputs.back(), 1, 1));
		m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, m_growthRate, 1, 1));
		m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_d_rightLayerOutputs.push_back(new iu::TensorGpu_32f(m_d_outRight->data(m_in*m_ih*m_iw * layerIdx*m_growthRate), m_in, m_growthRate, m_ih, m_iw, true, iu::TensorGpu_32f::NCHW));
		m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_growthRate, m_ih, m_iw, m_growthRate, m_growthRate, 3, 3, m_d_rightLayerOutputs.back(), 1, 1));
		m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, m_growthRate, 1, 1));
		m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));
	}

	// common path
	cudnnTensorDescriptor_t leftConcatDesc;
	cudnnCreateTensorDescriptor(&leftConcatDesc);
	cudnnSetTensor4dDescriptor(leftConcatDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, m_in, m_ic, m_ih, m_iw);

	std::vector<cudnnTensorDescriptor_t> corrInputs = { leftConcatDesc, leftConcatDesc };
    m_lrOps.push_back(new StereoCorrelation(corrInputs, d_unaryOut, min_disp, max_disp, step,rect_corr));
	m_lrOps.push_back(new Softmax(m_lrOps.back()->outTensorDesc(), CUDNN_SOFTMAX_MODE_CHANNEL));

	//m_lrOps.push_back(new SlackProp(m_lrOps.back()->outTensorDesc()));

	// pairwise net
	if(d_pairwiseOut != NULL)
	{
		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps[0]->outTensorDesc(), 2, 100, 3, 3, d_pairwiseOut));
		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));
	}

	m_initialized = true;
}
