#pragma once
#include "operator.h"

class Softmax: public Operator
{
public:
	Softmax(cudnnTensorDescriptor_t inOutTensorDesc, cudnnSoftmaxMode_t mode, cudnnSoftmaxAlgorithm_t algo=CUDNN_SOFTMAX_ACCURATE);
	~Softmax();

	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle);

private:
	cudnnSoftmaxAlgorithm_t m_softmaxAlgorithm;
	cudnnSoftmaxMode_t m_softmaxMode;

};
