#pragma once
#include <vector>
#include "cudnn.h"

#include "iu/iucore.h"

#include "operator.h"

class Activation : public Operator
{
public:
    Activation(cudnnTensorDescriptor_t inOutTensorDesc, cudnnActivationMode_t activationMode, double reluCeiling=0.);
	~Activation();

	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle);

private:
	// no copies!
	Activation(Activation const&);
	void operator=(Activation const&);

    cudnnActivationDescriptor_t m_activationDescriptor;

};
