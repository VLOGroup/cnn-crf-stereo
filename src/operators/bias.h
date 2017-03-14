#pragma once
#include "iu/iucore.h"

#include "cudnn.h"
#include "operator.h"
#include "../error_util.h"

class Bias : public Operator
{
public:
	Bias(cudnnTensorDescriptor_t inTensorDesc, cudnnTensorDescriptor_t biasTensorDesc);
	//Bias(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, cudnnTensorDescriptor_t biasTensorDesc);
	Bias(cudnnTensorDescriptor_t inTensorDesc, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw);
	//Bias(unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw);
	~Bias();

	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle);

	iu::TensorGpu_32f *getParams() { return m_bias; }

private:
	// no copies!
	Bias(Bias const&);
	void operator=(Bias const&);

	void initialize();

	iu::TensorGpu_32f *m_bias;
	cudnnTensorDescriptor_t m_biasTensorDesc;
	bool m_deleteBiasDesc;
};
