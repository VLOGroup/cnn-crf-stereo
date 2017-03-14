#pragma once
#include <cudnn.h>
#include "iu/iucore.h"

#include "operator.h"


class Convolution : public Operator
{
public:
	Convolution(cudnnConvolutionMode_t convMode, cudnnTensorDescriptor_t inTensorDesc, cudnnFilterDescriptor_t filterDesc, iu::TensorGpu_32f *d_out=NULL, int pad_x=0, int pad_y=0);
	Convolution(cudnnConvolutionMode_t convMode, cudnnTensorDescriptor_t inTensorDesc, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw, iu::TensorGpu_32f *d_out=NULL, int pad_x=0, int pad_y=0);
	Convolution(cudnnConvolutionMode_t convMode, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, unsigned int fk, unsigned int fc, unsigned int fh, unsigned int fw, iu::TensorGpu_32f *d_out=NULL, int pad_x=0, int pad_y=0);
	//Convolution(cudnnConvolutionMode_t convMode, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw, cudnnFilterDescriptor_t filterDesc);
	~Convolution();

	void deleteOutputMemory();

	iu::TensorGpu_32f *forward(std::vector<iu::TensorGpu_32f *> &d_inputs, cudnnHandle_t cudnnHandle);

	// getter / setter
	iu::TensorGpu_32f *getOutput() { return m_output; }
	cudnnTensorDescriptor_t outTensorDesc();
	iu::TensorGpu_32f *getParams() { return m_filter; }

private:
	// no copies!
	Convolution(Convolution const&);
	void operator=(Convolution const&);

	void initialize(iu::TensorGpu_32f *d_out, int pad_x, int pad_y);
	void allocateOutputMemory(iu::TensorGpu_32f *d_out);

	// cudnn conv settings
	cudnnConvolutionDescriptor_t m_convDesc;
	cudnnConvolutionMode_t m_convMode;

	//cudnnTensorFormat_t tensorFormat;

	// filter of this convolution
	iu::TensorGpu_32f *m_filter;
	cudnnFilterDescriptor_t m_filterDesc;
	bool m_deleteFilterDesc;

	// output
	bool m_extOutPtr;
	iu::TensorGpu_32f *m_output;
	cudnnTensorDescriptor_t m_outTensorDesc;
};
