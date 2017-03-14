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
