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

