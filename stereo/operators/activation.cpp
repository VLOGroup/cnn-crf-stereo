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

#include "activation.h"
#include "../error_util.h"

Activation::Activation(cudnnTensorDescriptor_t inOutTensorDesc, cudnnActivationMode_t activationMode, double reluCeiling) :
    Operator(inOutTensorDesc, Operator::Type::ACTIVATION)
{
    cudnnSafeCall(cudnnCreateActivationDescriptor(&m_activationDescriptor));
    cudnnSafeCall(cudnnSetActivationDescriptor(m_activationDescriptor,activationMode,CUDNN_NOT_PROPAGATE_NAN,reluCeiling));
}

Activation::~Activation()
{
	cudnnSafeCall(cudnnDestroyActivationDescriptor(m_activationDescriptor));
}

iu::TensorGpu_32f *Activation::forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle)
{
	float alpha = 1.0f;
	float beta = 0.0f;
    cudnnSafeCall(cudnnActivationForward(cudnnHandle, m_activationDescriptor, &alpha, outTensorDesc(),
				d_inputs[0]->data(), &beta, outTensorDesc(), d_inputs[0]->data()));

	return d_inputs[0];
}

