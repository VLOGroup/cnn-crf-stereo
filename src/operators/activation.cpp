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

