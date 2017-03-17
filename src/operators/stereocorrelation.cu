#include "iu/iucore.h"
#include "iu/ndarray/ndarray_ref.host.h"
#define divup(x,y) ((x-1)/(y)+1)

// get y for position x
__device__ float dLinearInterpolation1D(float x, float x0, float x1, float y0, float y1)
{
	return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

/*************************************************************************************************/
/* FORWARD Kernes                                                                                */
/*************************************************************************************************/
__global__ void kCorrelation_NHWC_interpolation(iu::TensorGpu_32f::TensorKernelData outNHWC,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outNHWC.length_)
		return;

	short n, d, h, w;
	outNHWC.coords(t_idx, &n, &h, &w, &d);

	float res = 0.0f;
	if (w + disparities(d) >= 0)
	{
		for (int c = 0; c < im0.C; ++c)
		{
			//res += im0(n, c, h, w) * im1(n, c, h, w + (int)disparities(d));

			// interpolation
			int floor_w = (int) floorf(w + disparities(d));
			int ceil_w = (int) (w + disparities(d) + 1);
			float w2 = w + disparities(d);

			res += im0(n, c, h, w)
					* dLinearInterpolation1D(w2, floor_w, ceil_w, im1(n, c, h, floor_w),
							im1(n, c, h, ceil_w));
		}
	}

	outNHWC(n, h, w, d) = res;
}

__global__ void kCorrelation_NHWC_interpolation(iu::TensorGpu_32f::TensorKernelData outNHWC,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities, int rectCorr)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outNHWC.length_)
		return;

	short n, d, h, w;
	outNHWC.coords(t_idx, &n, &h, &w, &d);

	outNHWC(n, h, w, d) = -100000;
	for (int y = -rectCorr; y <= rectCorr; ++y) // imperfect evaluation
	{
		if (h + y < 0 || h + y >= outNHWC.H)
			continue;

		float res = 0.0f;
		if (w + disparities(d) >= 0)
		{
			for (int c = 0; c < im0.C; ++c)
			{
				//res += im0(n, c, h, w) * im1(n, c, h + y, w + (int) disparities(d));

				// interpolation
				int floor_w = (int) floorf(w + disparities(d));
				int ceil_w = (int) (w + disparities(d) + 1);
				float w2 = w + disparities(d);

				res += im0(n, c, h, w)
						* dLinearInterpolation1D(w2, floor_w, ceil_w, im1(n, c, h + y, floor_w),
								im1(n, c, h + y, ceil_w));
			}
			if (res > outNHWC(n, h, w, d))
				outNHWC(n, h, w, d) = res;
		}
		else
		{
			// uniform distribution if outside the image
			outNHWC(n, h, w, d) = 0.0;//1.0 / outNHWC.C;
		}
	}
}

__global__ void kCorrelation_NHWC(iu::TensorGpu_32f::TensorKernelData outNHWC,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities, int rectCorr)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outNHWC.length_)
		return;

	short n, d, h, w;
	outNHWC.coords(t_idx, &n, &h, &w, &d);

	outNHWC(n, h, w, d) = -100000;
	for (int y = -rectCorr; y <= rectCorr; ++y) // imperfect evaluation
	{
		if (h + y < 0 || h + y >= outNHWC.H)
			continue;

		float res = 0.0f;
		if (w + disparities(d) >= 0)
		{
			for (int c = 0; c < im0.C; ++c)
			{
				res += im0(n, c, h, w) * im1(n, c, h + y, w + (int) disparities(d));
			}
		}

		if (res > outNHWC(n, h, w, d))
			outNHWC(n, h, w, d) = res;
	}

}
// prototype
// im0 [C x H x W]
// out [H x W x D]
__global__ void kCorrelation_1(kernel::ndarray_ref<float,3> out, kernel::ndarray_ref<float,3> im0, kernel::ndarray_ref<float,3> im1, kernel::ndarray_ref<float,1> disparities, int rectCorr){
	int channels = im0.size(0);
	int height = im0.size(1);
	int width = im0.size(2);
	int D = out.size(2);
	int w = threadIdx.x + blockIdx.x*blockDim.x;
	int h = threadIdx.y + blockIdx.y*blockDim.y;
	if(h >= height || w >= width ) return;
	//
	for(int d = 0;  d < D; ++d){
		int w2 = (int) disparities(d);
		if (w2 < 0 || w2 >= width) continue;
		float out_r = -100000;
		for (int y = -rectCorr; y <= rectCorr; ++y){
			int h2 = h + y;
			if (h2 < 0 || h2 >= height) continue;
			float res = 0.0f;
			for (int c = 0; c < channels; ++c){
				res += im0(c, h, w) * im1(c, h2, w2);
			}
			if (res > out_r)
				out_r = res;
		}
		out(h, w, d) = out_r;
	}
}

// can be used with or without imperfect rectified images
__global__ void kCorrelation_NCHW(iu::TensorGpu_32f::TensorKernelData outNCHW,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities, int rectCorr)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outNCHW.length_)
		return;

	short n, d, h, w;
	outNCHW.coords(t_idx, &n, &d, &h, &w);

	outNCHW(n, d, h, w) = -1000000;
	for(int y = -rectCorr; y <= rectCorr; ++y)
	{
		if (h + y < 0 || h + y >= outNCHW.H)
			continue;

		float res = 0.0f;
		if (w + disparities(d) >= 0)
		{
			for (int c = 0; c < im0.C; ++c)
			{
				res += im0(n, c, h, w) * im1(n, c, h + y, w + (int) disparities(d));
			}
		}

		if(res > outNCHW(n, d, h, w))
			outNCHW(n, d, h, w) = res;
	}

}

/*************************************************************************************************/
/* BACKWARD Kernels                                                                                */
/*************************************************************************************************/
__global__ void kCorrelationGrad_NCHW(iu::TensorGpu_32f::TensorKernelData outGrad0,
		iu::TensorGpu_32f::TensorKernelData outGrad1, iu::TensorGpu_32f::TensorKernelData inGrad,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outGrad0.length_)
		return;

	short n, c, h, w;
	outGrad0.coords(t_idx, &n, &c, &h, &w);

	// writing to these intermediate variables is twice as fast as summing up in global memory
	float grad0 = 0.0f;
	float grad1 = 0.0f;
	for (int d = 0; d < inGrad.C; ++d)
	{
		if ((w + disparities(d)) >= 0 && (w + disparities(d)) < outGrad0.W)
		{
			grad0 += im1(n, c, h, w + disparities(d)) * inGrad(n, d, h, w);
		}
		if ((w - disparities(d)) >= 0 && (w - disparities(d)) < outGrad0.W)
		{
			grad1 += im0(n, c, h, w - disparities(d)) * inGrad(n, d, h, w - disparities(d));
		}
	}

	outGrad0(n, c, h, w) = grad0;
	outGrad1(n, c, h, w) = grad1;
}

// TODO: Preload inGrad for this pixel into shared memory
__global__ void kCorrelationGrad_NHWC(iu::TensorGpu_32f::TensorKernelData outGrad0,
		iu::TensorGpu_32f::TensorKernelData outGrad1, iu::TensorGpu_32f::TensorKernelData inGrad,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities)
{
	int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (t_idx >= outGrad0.length_)
		return;

	// threads are started for each output element (coords(im0) = coords(im1))
	short n, c, h, w;
	im0.coords(t_idx, &n, &c, &h, &w);

	// writing to these intermediate variables is twice as fast as summing up in global memory
	float grad0 = 0.0f;
	float grad1 = 0.0f;
	for (int d = 0; d < inGrad.C; ++d)
	{
		if ((w + disparities(d)) >= 0 && (w + disparities(d)) < outGrad0.W)
		{
			grad0 += im1(n, c, h, w + disparities(d)) * inGrad(n, h, w, d);
		}
	}

	for (int d = 0; d < inGrad.C; ++d)
	{
		if ((w - disparities(d)) >= 0 && (w - disparities(d)) < outGrad0.W)
		{
			grad1 += im0(n, c, h, w - disparities(d)) * inGrad(n, h, w - disparities(d), d);
		}
	}

	outGrad0(n, c, h, w) = grad0;
	outGrad1(n, c, h, w) = grad1;
}

__device__ float parallelReductionSum(float value, volatile float *shared_buffer, int linId)
{
	shared_buffer[linId] = value;
	__syncthreads();


	if(blockDim.x > 128 && linId + 128 < blockDim.x)
	{
		shared_buffer[linId] += shared_buffer[linId + 128];
	}
	__syncthreads();

	if(blockDim.x > 64 && linId + 64 < blockDim.x)
	{
		shared_buffer[linId] += shared_buffer[linId + 64];
	}
	__syncthreads();
	if(linId < 32)
	{
		shared_buffer[linId] += shared_buffer[linId + 32];
		shared_buffer[linId] += shared_buffer[linId + 16];
		shared_buffer[linId] += shared_buffer[linId + 8];
		shared_buffer[linId] += shared_buffer[linId + 4];
		shared_buffer[linId] += shared_buffer[linId + 2];
		shared_buffer[linId] += shared_buffer[linId + 1];
	}
	__syncthreads();
	if(linId == 0)
		return shared_buffer[0];
	else
		return -1000;
}

__global__ void kCorrelationGrad_NHWC_shared(iu::TensorGpu_32f::TensorKernelData outGrad0,
		iu::TensorGpu_32f::TensorKernelData outGrad1, iu::TensorGpu_32f::TensorKernelData inGrad,
		iu::TensorGpu_32f::TensorKernelData im0, iu::TensorGpu_32f::TensorKernelData im1,
		iu::LinearDeviceMemory_32f_C1::KernelData disparities)
{

	// threads are started for each output element (coords(im0) = coords(im1))
	short w = blockIdx.x;
	short h = blockIdx.y;
	short c = blockIdx.z;
	short d = threadIdx.x;

	extern __shared__ float buffer[];

	// load data
	float grad0 = 0.0f;
	float grad1 = 0.0f;

	// grad 0
	if ((w + disparities(d)) >= 0 && (w + disparities(d)) < outGrad0.W)
	{
		grad0 = im1(0, c, h, w + disparities(d)) * inGrad(0, h, w, d);
	}
	float res = parallelReductionSum(grad0, buffer, d);
	if(d == 0)
		outGrad0(0, c, h, w) = res;
	//__syncthreads();

	// grad1
	if ((w - disparities(d)) >= 0 && (w - disparities(d)) < outGrad0.W)
	{
		grad1 = im0(0, c, h, w - disparities(d)) * inGrad(0, h, w - disparities(d), d);
	}

	res = parallelReductionSum(grad1, buffer, d);
	if(d == 0)
		outGrad1(0, c, h, w) = res;
}

/*************************************************************************************************/
/* Kernel Callers                                                                                */
/*************************************************************************************************/
namespace cuda
{

void forward(iu::TensorGpu_32f &d_out, iu::TensorGpu_32f &d_inLeft, iu::TensorGpu_32f &d_inRight,
		iu::LinearDeviceMemory_32f_C1 &d_disparities, int rectCorr)
{
//	iu::IuCudaTimer cut;
//	cut.start();

	// check float or int disparities
	bool dispIsFloat = false;
	iu::LinearHostMemory_32f_C1 h_disparities(d_disparities.size());
	iu::copy(&d_disparities, &h_disparities);

	float eps = 0.001;
	if(std::abs(h_disparities.getPixel(1) - std::round(h_disparities.getPixel(1))) > eps)
	{
		dispIsFloat = true;
	}


	//dim3 threadsPerBlock(16 * 16);
	dim3 threadsPerBlock(480);
	dim3 numBlocks(
			std::ceil(
					(d_out.samples() * d_out.height() * d_out.width() * d_out.channels())
							/ static_cast<float>(threadsPerBlock.x)));

	if (d_out.memoryLayout() == iu::TensorGpu_32f::NCHW)
	{
		kCorrelation_NCHW<<<numBlocks, threadsPerBlock>>>(d_out, d_inLeft, d_inRight, d_disparities, rectCorr);

		if(dispIsFloat)
			std::cout << "Warning: You are calling a kernel with NO interpolation with float disparities!" << std::endl;

	}
	else if(d_out.memoryLayout() == iu::TensorGpu_32f::NHWC)
	{
		//rectCorr = 1.0;
		if(!dispIsFloat)
			kCorrelation_NHWC<<<numBlocks, threadsPerBlock>>>(d_out, d_inLeft, d_inRight, d_disparities, rectCorr);
		else
		{
			kCorrelation_NHWC_interpolation<<<numBlocks, threadsPerBlock>>>(d_out, d_inLeft, d_inRight, d_disparities, rectCorr);
			std::cout << "Call interpolation-kernel" << std::endl;
		}
	}
	else
        std::cout << "[StereoCorrelation] no cuda code called" << std::endl;

	cudaDeviceSynchronize();

//	std::cout << cut.elapsed() << std::endl;
}

// d_inLeft : [S x C x H x W]
// d_out    : [S x H x W x D]
void forward_1(ndarray_ref<float,4> &d_out, const ndarray_ref<float,4> &d_inLeft, const ndarray_ref<float,4> &d_inRight, const ndarray_ref<float,1> &d_disparities, int rectCorr)
{
	dim3 threadsPerBlock(16,16);
	int H = d_inLeft.size(2);
	int W = d_inLeft.size(3);
	dim3 numBlocks(divup(W,threadsPerBlock.x),divup(H,threadsPerBlock.y));
	//
	for(int n=0; n < d_inLeft.size(0); ++n){
		kCorrelation_1<<<numBlocks, threadsPerBlock>>>(d_out.subdim<0>(n), d_inLeft.subdim<0>(n), d_inRight.subdim<0>(n), d_disparities, rectCorr);
	}
	cudaDeviceSynchronize();
}


void forward(float *d_out, float *d_inLeft, float *d_inRight, float *d_disparities, int in, int ic,
		int ih, int iw, int numDisps, iu::TensorGpu_32f::MemoryLayout memoryLayout, int rectCorr)
{
	iu::TensorGpu_32f d_in0(d_inLeft, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
	iu::TensorGpu_32f d_in1(d_inRight, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
	iu::LinearDeviceMemory_32f_C1 d_disps(d_disparities, numDisps, true);
	iu::TensorGpu_32f d_output(d_out, in, numDisps, ih, iw, true, memoryLayout);

	forward(d_output, d_in0, d_in1, d_disps, rectCorr);
}

void backward(float *d_outGrad0, float *d_outGrad1, float *d_inGrad, float *d_im0, float *d_im1,
		float *d_disparities, int in, int ic, int ih, int iw, int numDisps,
		iu::TensorGpu_32f::MemoryLayout memoryLayout)
{
	iu::TensorGpu_32f d_inIm0(d_im0, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
	iu::TensorGpu_32f d_inIm1(d_im1, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);

	iu::TensorGpu_32f d_inGradient(d_inGrad, in, numDisps, ih, iw, true, memoryLayout);
	iu::TensorGpu_32f d_outGradient0(d_outGrad0, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);
	iu::TensorGpu_32f d_outGradient1(d_outGrad1, in, ic, ih, iw, true, iu::TensorGpu_32f::NCHW);

	iu::LinearDeviceMemory_32f_C1 d_disps(d_disparities, numDisps, true);

	dim3 threadsPerBlock(16 * 16);
	dim3 numBlocks(std::ceil((in * ic * ih * iw) / static_cast<float>(threadsPerBlock.x)));

	if (d_inGradient.memoryLayout() == iu::TensorGpu_32f::NCHW)
	{
		kCorrelationGrad_NCHW<<<numBlocks, threadsPerBlock>>>(d_outGradient0, d_outGradient1, d_inGradient, d_inIm0, d_inIm1, d_disps);
	}
	else if(d_inGradient.memoryLayout() == iu::TensorGpu_32f::NHWC)
	{
#if 0
			kCorrelationGrad_NHWC<<<numBlocks, threadsPerBlock>>>(d_outGradient0, d_outGradient1, d_inGradient, d_inIm0, d_inIm1, d_disps);
			std::cout << cut.elapsed() << std::endl;
#else

			//cut.start();
			dim3 threadsPerBlock_shared(numDisps);
			dim3 numBlocks_shared(iw, ih, ic);
			kCorrelationGrad_NHWC_shared<<<numBlocks_shared, threadsPerBlock_shared, numDisps * sizeof(float)>>>(d_outGradient0, d_outGradient1, d_inGradient, d_inIm0, d_inIm1, d_disps);
#endif
	}
	else
		std::cout << "[StereoCorrelation] no cuda backward code called" << std::endl;

	cudaDeviceSynchronize();
}

}
