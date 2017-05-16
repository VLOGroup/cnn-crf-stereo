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

#include "softmax.h"
#include "../error_util.h"

Softmax::Softmax(cudnnTensorDescriptor_t inOutTensorDesc, cudnnSoftmaxMode_t mode, cudnnSoftmaxAlgorithm_t algo) :
		Operator(inOutTensorDesc, Operator::Type::SOFTMAX), m_softmaxMode(mode), m_softmaxAlgorithm(algo)
{
}

Softmax::~Softmax()
{
}

iu::TensorGpu_32f *Softmax::forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle)
{
	float alpha = 1.0;
	float beta = 0.0;
	cudnnSafeCall(
			cudnnSoftmaxForward(cudnnHandle, m_softmaxAlgorithm, m_softmaxMode, &alpha, m_inTensorDesc[0],
					d_inputs[0]->data(), &beta, m_inTensorDesc[0], d_inputs[0]->data()));

	return d_inputs[0];
}
