#include "cudnn.h"

#include "../error_util.h"
#include "stereocorrelation.h"
#include "stereocorrelation.cuh"

#include <ctime>

StereoCorrelation::StereoCorrelation(std::vector<cudnnTensorDescriptor_t> &inTensorDesc, iu::TensorGpu_32f *d_out, float min_disp, float max_disp, float step, int rect_corr) :
        Operator(inTensorDesc, Operator::Type::STEREO_CORRELATION), m_disparities(NULL), m_output(NULL), m_outTensorDesc(NULL), m_rectCorr(rect_corr)
{
	cudnnSafeCall(cudnnCreateTensorDescriptor(&m_outTensorDesc));

    if(d_out) {
		m_extOutPtr = true;
        setDisparities(min_disp, max_disp, step, d_out);
    } else {
		m_extOutPtr = false;
        setDisparities(0.0, 127.0, 1.0, d_out);
    }
}

StereoCorrelation::~StereoCorrelation()
{
	cudnnSafeCall(cudnnDestroyTensorDescriptor(m_outTensorDesc));
	if (!m_extOutPtr)
		if (m_output)
			delete m_output;

	if (m_disparities)
		delete m_disparities;
}

iu::TensorGpu_32f *StereoCorrelation::forward(std::vector<iu::TensorGpu_32f *> &d_inputs,
		cudnnHandle_t cudnnHandle)
{
	if (d_inputs.size() != 2)
	{
		std::cerr << "Error: StereoCorrelation needs exactly two device inputs!" << std::endl;
		exit(-1);
	}

	// do not use rectification correction here
    cuda::forward(*m_output, *(d_inputs[0]), *(d_inputs[1]), *m_disparities, m_rectCorr);

	return m_output;
}

cudnnTensorDescriptor_t StereoCorrelation::outTensorDesc()
{
	return m_outTensorDesc;
}

void StereoCorrelation::setDisparities(float min_disp, float max_disp, float step, iu::TensorGpu_32f *d_out)
{
	// some cleanup
	if (m_disparities != NULL)
	{
        delete m_disparities;
    }
    // if we previously had an internal data pointer, we free it.
    if(!m_extOutPtr)
        delete m_output;

	// disparities
	std::vector<float> disparities;
	for (float d = min_disp; d <= max_disp; d += step)
		disparities.push_back(-d);

	//PixelType* host_data, const unsigned int& length, bool ext_data_pointer = false
	iu::LinearHostMemory_32f_C1 disps(&(disparities[0]), disparities.size(), true);

	m_disparities = new iu::LinearDeviceMemory_32f_C1(disparities.size());
	iu::copy(&disps, m_disparities);

	// out tensor descriptor
	int n, c, h, w, nStride, cStride, hStride, wStride;
	cudnnDataType_t dataType;
	cudnnSafeCall(
			cudnnGetTensor4dDescriptor(m_inTensorDesc[0], &dataType, &n, &c, &h, &w, &nStride,
					&cStride, &hStride, &wStride));

	// output memory
	if(d_out != NULL)
	{
		// check size
		// TODO: cleanup if something is wrong
		if(d_out->samples() != n)
		{
			std::cerr << "Error: Size of provided cost-volume is not correct! num-samples do not match! Given size = " << d_out->samples() << " should be " << n << std::endl;
			exit(-1);
		}
		if(d_out->channels() != disparities.size())
		{
			std::cerr << "Error: Size of provided cost-volume is not correct! disparities do not match! Given size = " << d_out->channels() << " should be " << disparities.size() << std::endl;
			exit(-1);
		}
		if(d_out->height() != h)
		{
			std::cerr << "Error: Size of provided cost-volume is not correct! height does not match! Given size = " << d_out->height() << " should be " << h << std::endl;
			exit(-1);
		}
		if(d_out->width() != w)
		{
			std::cerr << "Error: Size of provided cost-volume is not correct! weight does not match! Given size = " << d_out->width() << " should be " << w << std::endl;
			exit(-1);
		}
		if(d_out->memoryLayout() != iu::TensorGpu_32f::MemoryLayout::NHWC)
		{
			std::cerr << "Error: Provided cost-volume has wrong memory layout!" << std::endl;
			exit(-1);
		}

		m_output = d_out;
		m_extOutPtr = true;
	}
	else
	{
		m_output = new iu::TensorGpu_32f(n, disparities.size(), h, w,
				iu::TensorGpu_32f::MemoryLayout::NHWC);
		m_extOutPtr = false;
	}

	cudnnSafeCall(
			cudnnSetTensor4dDescriptor(m_outTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n,
					disparities.size(), h, w));
}

void StereoCorrelation::forward(float *d_input0, float *d_input1, float *d_disparities,
		float *d_output, int in, int ic, int ih, int iw, int lenDisps,
		iu::TensorGpu_32f::MemoryLayout memoryLayout, int rectCorr)
{
	cuda::forward(d_output, d_input0, d_input1, d_disparities, in, ic, ih, iw, lenDisps,
			memoryLayout, rectCorr);
}

void StereoCorrelation::backward(float *d_outGrad0, float *d_outGrad1, float *d_inGrad,
		float *d_im0, float *d_im1, float *d_disparities, int n, int c, int h, int w, int numDisps,
		iu::TensorGpu_32f::MemoryLayout memoryLayout)
{
	cuda::backward(d_outGrad0, d_outGrad1, d_inGrad, d_im0, d_im1, d_disparities, n, c, h, w,
			numDisps, memoryLayout);
}
