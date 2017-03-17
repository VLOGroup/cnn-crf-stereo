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
#include "time.h"

#include "cnpy.h"
#include "utils.h"

#include "stereonet.h"
#include "operators/stereocorrelation.h"
#include "operators/convolution.h"
#include "operators/activation.h"
#include "operators/softmax.h"
#include "operators/bias.h"
#include "operators/slackprop.h"

#include <iu/iumath.h>

StereoNet::StereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw) :
		m_numLayers(numLayers), m_in(in), m_ic(ic), m_ih(ih), m_iw(iw), m_verbose(false), m_initialized(false),
		m_pwInput(NULL), m_allowGc(false)
{
	cudnnSafeCall(cudnnCreate(&m_cudnnHandle));
}

StereoNet::~StereoNet()
{
	cudnnSafeCall(cudnnDestroy(m_cudnnHandle));

	for (auto op : m_leftOps)
		if(op)
			delete op;

	for (auto op : m_rightOps)
		if(op)
			delete op;

	for (auto op : m_lrOps)
		if(op)
			delete op;

	for (auto op : m_pairwiseOps)
		if(op)
			delete op;

	if(m_pwInput)
		delete m_pwInput;
}

void StereoNet::initNet(float min_disp, float max_disp, float step, int rect_corr,
						iu::TensorGpu_32f *d_unaryOut, iu::TensorGpu_32f *d_pairwiseOut)
{
	// left path
	m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_ic, m_ih, m_iw, 100, m_ic, 3, 3));
	m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
	m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
	m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, m_ic, m_ih, m_iw, 100, m_ic, 3, 3));
	m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
	m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
	m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
	m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

	if (m_numLayers > 3)
	{
		// left
		m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		// right
		m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));
	}

	if (m_numLayers > 5)
	{
		// left
		m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_leftOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_leftOps.push_back(new Bias(m_leftOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_leftOps.push_back(new Activation(m_leftOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		// right
		m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_rightOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_rightOps.back()->outTensorDesc(), 100, 100, 2, 2));
		m_rightOps.push_back(new Bias(m_rightOps.back()->outTensorDesc(), 1, 100, 1, 1));
		m_rightOps.push_back(new Activation(m_rightOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));
	}

	// common path
	std::vector<cudnnTensorDescriptor_t> corrInputs = { m_leftOps.back()->outTensorDesc(),
			m_rightOps.back()->outTensorDesc() };
    m_lrOps.push_back(new StereoCorrelation(corrInputs, d_unaryOut, min_disp, max_disp, step, rect_corr));
	m_lrOps.push_back(new Softmax(m_lrOps.back()->outTensorDesc(), CUDNN_SOFTMAX_MODE_CHANNEL));

	//m_lrOps.push_back(new SlackProp(m_lrOps.back()->outTensorDesc()));

	// pairwise net
	if(d_pairwiseOut != NULL)
	{
//		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_leftOps[0]->outTensorDesc(), 2, 100, 3, 3, d_pairwiseOut));
//		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		int ih = m_ih;
		int iw = m_iw;
		if(m_numLayers == 7)
		{
			ih = m_ih - 4;
			iw = m_iw - 4;
		}

		if(m_verbose)
			std::cout << "construct pairwise ops" << std::endl;
		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_in, 3, ih, iw, 64, 3, 3, 3));
		m_pairwiseOps.push_back(new Bias(m_pairwiseOps.back()->outTensorDesc(), 1, 64, 1, 1));
		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_pairwiseOps.back()->outTensorDesc(), 64, 64, 3, 3));
		m_pairwiseOps.push_back(new Bias(m_pairwiseOps.back()->outTensorDesc(), 1, 64, 1, 1));
		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_TANH));

		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_pairwiseOps.back()->outTensorDesc(), 2, 64, 1, 1, d_pairwiseOut));

//		m_pairwiseOps.push_back(new Convolution(CUDNN_CROSS_CORRELATION, m_in, m_ic, ih, iw, 64, m_ic, 3, 3));
//		m_pairwiseOps.push_back(new Bias(m_pairwiseOps.back()->outTensorDesc(), 1, 64, 1, 1));
//		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_RELU));
//
//		m_pairwiseOps.push_back(new Convolution(CUDNN_CROSS_CORRELATION, m_pairwiseOps.back()->outTensorDesc(), 64, 64, 3, 3));
//		m_pairwiseOps.push_back(new Bias(m_pairwiseOps.back()->outTensorDesc(), 1, 64, 1, 1));
//		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_RELU));
//
//		m_pairwiseOps.push_back(new Convolution(CUDNN_CONVOLUTION, m_pairwiseOps.back()->outTensorDesc(), 2, 64, 1, 1, d_pairwiseOut));
//		m_pairwiseOps.push_back(new Activation(m_pairwiseOps.back()->outTensorDesc(), CUDNN_ACTIVATION_RELU));
	}

	m_initialized = true;
}

void StereoNet::setAllParams(std::string npzPath)
{
	assert(m_leftOps.size() == m_rightOps.size());

//	m_verbose=true;

	// load from npz array
	int param_idx = 0;
	for (int op_idx = 0; op_idx < m_leftOps.size(); ++op_idx)
	{
		if (m_leftOps[op_idx]->type() == Operator::Type::ACTIVATION)
			continue;

		if(m_verbose)
			std::cout << "Unary: params_idx " << param_idx << std::endl;
		std::string key = "arr_";
		key.append(std::to_string(param_idx));
		m_leftOps[op_idx]->setParams(npzPath, key);
		m_rightOps[op_idx]->setParams(npzPath, key);

//		std::cout << "set param " << param_idx << std::endl;

		param_idx += 1;
	}

	if (m_verbose)
		std::cout << "Loaded unary-params successfully" << std::endl;

	for(int op_idx = 0; op_idx < m_pairwiseOps.size(); ++op_idx)
	{
		if (m_pairwiseOps[op_idx]->type() == Operator::Type::ACTIVATION)
			continue;

		if(m_verbose)
			std::cout << "PW: params_idx " << param_idx << std::endl;
		std::string key = "arr_";
		key.append(std::to_string(param_idx));
		m_pairwiseOps[op_idx]->setParams(npzPath, key);

		param_idx += 1;
	}

	if (m_verbose)
		std::cout << "Loaded pairwise-params successfully" << std::endl;
}

void StereoNet::setDisparities(float min_disp, float max_disp, float step, iu::TensorGpu_32f *d_out)
{
	if(m_lrOps[0]->type() != Operator::STEREO_CORRELATION)
	{
		std::cout << "Error: Could not find Stereo Correlation Operation" << std::endl;
		exit(-1);
	}

	StereoCorrelation *corr = dynamic_cast<StereoCorrelation*>(m_lrOps[0]);
	if(corr)
	{
		// downcast successful
		corr->setDisparities(min_disp, max_disp, step, d_out);
	}
	else
	{
		std::cout << "Error: Could not downcast to Stereo Correlation" << std::endl;
		exit(-1);
	}
}

iu::TensorGpu_32f *StereoNet::performPrediction(iu::TensorGpu_32f *d_inputLeft, iu::TensorGpu_32f *d_inputRight)
{
	if(!m_initialized)
	{
		std::cout << "Error: Net is not initialized!" << std::endl;
		exit(-1);
	}

	iu::IuCudaTimer cut;

//	// compute wx, wy
//	float lambda = 0.4;
//	float alpha = -20.0;
//	float beta = 0.9;
//	iu::TensorGpu_32f d_gradX(1, 1, d_inputLeft.height(), d_inputLeft.width());
//	cuda::gradX(d_inputLeft, d_gradX);
//	cuda::expTensor(d_gradX, alpha, beta, lambda);
//
//	iu::TensorGpu_32f d_gradY(1, 1, d_inputLeft.height(), d_inputLeft.width());
//	cuda::gradY(d_inputLeft, d_gradY);
//	cuda::expTensor(d_gradY, alpha, beta, lambda);


	if (m_verbose)
		cut.start();

	if(m_verbose)
		std::cout << "Execute pairwise net" << std::endl;

	// perform forward pass
	iu::TensorGpu_32f *d_outLeft = d_inputLeft;
	iu::TensorGpu_32f *d_outRight = d_inputRight;

	for (int op_idx = 0; op_idx < m_leftOps.size(); ++op_idx)
	{
		d_outLeft = m_leftOps[op_idx]->forward(d_outLeft, m_cudnnHandle);
	}

	for (int op_idx = 0; op_idx < m_leftOps.size(); ++op_idx)
	{
		d_outRight = m_rightOps[op_idx]->forward(d_outRight, m_cudnnHandle);
	}

	if (m_verbose)
	{
		std::cout << "Elapsed time (convolutions): " << cut.elapsed() << std::endl;
		cut.start();
	}

	iu::TensorGpu_32f *d_commonOut;
	std::vector<iu::TensorGpu_32f *> commonInputs { d_outLeft, d_outRight };
	d_commonOut = m_lrOps[0]->forward(commonInputs, m_cudnnHandle);

	if (m_verbose)
	{
		std::cout << "Elapsed time (correlation): " << cut.elapsed() << std::endl;
		cut.start();
	}

	d_commonOut = m_lrOps[1]->forward(d_commonOut, m_cudnnHandle);

	if (m_verbose)
	{
		std::cout << "Elapsed time (softmax): " << cut.elapsed() << std::endl;
		//	std::cout << "Elapsed time (correlation + softmax): " << cut.elapsed() << std::endl;
	}

	// convert to minimization
	cuda::negateTensor(*d_commonOut);

	if(m_allowGc == true)
	{
		// free output-buffer of conv-layers
		for(int idx = 0; idx < m_leftOps.size(); ++idx)
		{
			Convolution *leftConvOp = dynamic_cast<Convolution*>(m_leftOps[idx]);
			if(leftConvOp)
			{
				leftConvOp->deleteOutputMemory();
				if(m_verbose)
					std::cout << "Free left op idx " << idx << std::endl;
			}
			Convolution *rightConvOp = dynamic_cast<Convolution*>(m_rightOps[idx]);
			if(rightConvOp)
			{
				rightConvOp->deleteOutputMemory();
				if(m_verbose)
					std::cout << "Free right op idx " << idx << std::endl;

			}
		}
	}

	if(m_pairwiseOps.size() == 2) // reuse unary pairwise
	{
		// pairwise-net
		if (m_verbose)
			cut.start();

		if(m_pairwiseOps[0]->type() != Operator::CONVOLUTION)
		{
			std::cout << "Error: Could not find Pairwise Convolution Operation" << std::endl;
			exit(-1);
		}

		Convolution *outConv1 = dynamic_cast<Convolution*>(m_leftOps[0]);
		if(outConv1)
		{
			// downcast successful
			iu::TensorGpu_32f *d_outPw = m_pairwiseOps[0]->forward(outConv1->getOutput(), m_cudnnHandle);
			d_outPw = m_pairwiseOps[1]->forward(d_outPw, m_cudnnHandle);
		}
		else
		{
			std::cout << "Error: Could not downcast to Stereo Correlation" << std::endl;
			exit(-1);
		}

		if (m_verbose)
		{
			std::cout << "Elapsed time (pairwise-net): " << cut.elapsed() << std::endl;
		}
	}
	else if(m_pairwiseOps.size() > 2) // stand-alone pw-net
	{
        iu::IuCudaTimer cut;
        if(m_verbose) {
            cut.start();
			std::cout << "Execute pairwise net" << std::endl;
        }
		// initialize input (left image)
		iu::TensorGpu_32f *d_outPw = d_inputLeft;
		if(m_numLayers == 7)
		{
			if(m_pwInput != NULL)
			{
				d_outPw = m_pwInput;
				if(m_verbose)
					std::cout << "Use cropped pw input" << std::endl;
			}
			else
			{
				std::cout << "ERROR: Need different sized pw input, but does not exist yet!" << std::endl;
				exit(-1);
			}
		}

		for (int op_idx = 0; op_idx < m_pairwiseOps.size(); ++op_idx)
		{
			d_outPw = m_pairwiseOps[op_idx]->forward(d_outPw, m_cudnnHandle);
		}
		iu::math::abs(*d_outPw, *d_outPw);
        if(m_verbose)
            std::cout << "Elapsed time PW Net: " << cut.elapsed() << std::endl;
//		save(*d_outPw, "/tmp/dout.npy");
//		save(*d_commonOut, "/tmp/un.npy");
	}

	//m_lrOps[2]->forward()
	if(m_verbose)
		std::cout << "Finished prediction" << std::endl;


	return d_commonOut;

//	iu::TensorGpu_32f *d_spOut = new iu::TensorGpu_32f(1, 1, d_out->height(), d_out->width());
//	SlackProp::forward(d_out->data(), d_gradX.data(), d_gradY.data(), d_spOut->data(), 1, d_out->channels(), d_out->height(), d_out->width(), 1.0, 12.0);
//
//	delete d_out;
//
//	return d_spOut;
}
