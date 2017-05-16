// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
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

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <iostream>

#include <iu/iucore.h>

#include "utils.cuh"

#define divup(x,y) ((x-1)/(y)+1)

struct std_functor
{
	const float mu;

	std_functor(float mu) :
			mu(mu)
	{
	}

	__host__ __device__ float operator()(const float x) const
	{
		return (x - mu) * (x - mu);
	}
};

struct square
{
	__host__ __device__ float operator()(const float x) const
	{
		return x * x;
	}
};

struct ExpEdges
{
	float alpha;
	float beta;
	float lambda;

	ExpEdges(float alpha, float beta, float lambda) :	alpha(alpha), beta(beta), lambda(lambda)
	{
	}

	__host__ __device__ float operator()(const float x) const
	{
		return lambda * exp(alpha * pow(abs(x), beta));
	}
};

__global__ void grad_x(iu::TensorGpu_32f::TensorKernelData outGradX, iu::TensorGpu_32f::TensorKernelData inImg)
{
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if (   x_idx < 0 || x_idx >= inImg.W
		|| y_idx < 0 || y_idx >= inImg.H)
		return;

	if(x_idx + 1 == inImg.W)
		outGradX(0, 0, y_idx, x_idx) = 0.0;
	else
		outGradX(0, 0, y_idx, x_idx) = inImg(0, 0, y_idx, x_idx + 1) - inImg(0, 0, y_idx, x_idx);
}

__global__ void grad_y(iu::TensorGpu_32f::TensorKernelData outGradY, iu::TensorGpu_32f::TensorKernelData inImg)
{
	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if (   x_idx < 0 || x_idx >= inImg.W
		|| y_idx < 0 || y_idx >= inImg.H)
		return;

	if(y_idx + 1 == inImg.H)
		outGradY(0, 0, y_idx, x_idx) = 0.0;
	else
		outGradY(0, 0, y_idx, x_idx) = inImg(0, 0, y_idx + 1, x_idx) - inImg(0, 0, y_idx, x_idx);
}

__global__ void kConncatChannels(iu::TensorGpu_32f::TensorKernelData first, iu::TensorGpu_32f::TensorKernelData second, iu::TensorGpu_32f::TensorKernelData output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z * blockDim.z + threadIdx.z;

	if(x >= output.W || y >= output.H || c >= output.C)
		return;

	if(c < 3)
	{
		output(0, c, y, x) = first(0, c, y, x);
	}
	else
	{
		output(0, c, y, x) = second(0, c - 3, y, x);
	}


}

namespace cuda
{

float makeZeroMean(iu::ImageGpu_32f_C1 &d_inOut)
{
	float mu = cuda::mean(d_inOut);
	thrust::transform(d_inOut.begin(), d_inOut.end(), thrust::constant_iterator<float>(mu), d_inOut.begin(),
			thrust::minus<float>());

	return mu;
}

float makeUnitStd(iu::ImageGpu_32f_C1 &d_inOut)
{
	float std_val = cuda::std(d_inOut);
	thrust::transform(d_inOut.begin(), d_inOut.end(), thrust::constant_iterator<float>(std_val), d_inOut.begin(),
			thrust::divides<float>());

	return std_val;
}

float sum(iu::ImageGpu_32f_C1 &d_img)
{
	return thrust::reduce(d_img.begin(), d_img.end(), 0.0, thrust::plus<float>());
}

float mean(iu::ImageGpu_32f_C1 &d_img)
{
	int N = d_img.width() * d_img.height();
	return cuda::sum(d_img) / N;
}

float std(iu::ImageGpu_32f_C1 &d_img, float mean)
{
	int N = d_img.width() * d_img.height();

	if (mean == -11111111)
		mean = cuda::mean(d_img);

//	float eps = 1e-7;
//	if(mean > -eps && mean < eps)
//		return sqrt(thrust::transform_reduce(d_img.begin(), d_img.end(), square(), 0.0, thrust::plus<float>()) / (N - 1));
//	else
		return sqrt(
			thrust::transform_reduce(d_img.begin(), d_img.end(), std_functor(mean), 0.0, thrust::plus<float>())
					/ (N));
}

void makeRgbZeroMean(iu::TensorGpu_32f &d_inOut, float *out_r_mean, float *out_g_mean, float *out_b_mean)
{
	auto r_begin = d_inOut.begin();
	auto r_end = thrust::device_ptr<float>(d_inOut.data(d_inOut.width() * d_inOut.height()));
	auto g_begin = thrust::device_ptr<float>(d_inOut.data(d_inOut.width() * d_inOut.height()));;
	auto g_end = thrust::device_ptr<float>(d_inOut.data(2 * d_inOut.width() * d_inOut.height()));
	auto b_begin = thrust::device_ptr<float>(d_inOut.data(2 * d_inOut.width() * d_inOut.height()));
	auto b_end = d_inOut.end();

	int N = d_inOut.width() * d_inOut.height();
	float r_mean = thrust::reduce(r_begin, r_end, 0.0, thrust::plus<float>()) / N;
	float g_mean = thrust::reduce(g_begin, g_end, 0.0, thrust::plus<float>()) / N;
	float b_mean = thrust::reduce(b_begin, b_end, 0.0, thrust::plus<float>()) / N;

	thrust::transform(r_begin, r_end, thrust::constant_iterator<float>(r_mean), r_begin, thrust::minus<float>());
	thrust::transform(g_begin, g_end, thrust::constant_iterator<float>(g_mean), g_begin, thrust::minus<float>());
	thrust::transform(b_begin, b_end, thrust::constant_iterator<float>(b_mean), b_begin, thrust::minus<float>());

	if(out_r_mean != NULL)
		*out_r_mean = r_mean;
	if(out_g_mean != NULL)
		*out_g_mean = g_mean;
	if(out_b_mean != NULL)
		*out_b_mean = b_mean;
}

void makeRgbUnitStd(iu::TensorGpu_32f &d_inOut, bool isZeroMean, float *out_r_std, float *out_g_std, float *out_b_std)
{
	auto r_begin = d_inOut.begin();
	auto r_end = thrust::device_ptr<float>(d_inOut.data(d_inOut.width() * d_inOut.height()));
	auto g_begin = thrust::device_ptr<float>(d_inOut.data(d_inOut.width() * d_inOut.height()));;
	auto g_end = thrust::device_ptr<float>(d_inOut.data(2 * d_inOut.width() * d_inOut.height()));
	auto b_begin = thrust::device_ptr<float>(d_inOut.data(2 * d_inOut.width() * d_inOut.height()));
	auto b_end = d_inOut.end();

	int N = d_inOut.width() * d_inOut.height();
	float r_std = 0.0;
	float g_std = 0.0;
	float b_std = 0.0;

	if(!isZeroMean)
	{
		int N = d_inOut.width() * d_inOut.height();
		float r_mean = thrust::reduce(r_begin, r_end, 0.0, thrust::plus<float>()) / N;
		float g_mean = thrust::reduce(g_begin, g_end, 0.0, thrust::plus<float>()) / N;
		float b_mean = thrust::reduce(b_begin, b_end, 0.0, thrust::plus<float>()) / N;

		r_std = sqrt(thrust::transform_reduce(r_begin, r_end, std_functor(r_mean), 0.0, thrust::plus<float>()) / (N));
		g_std = sqrt(thrust::transform_reduce(g_begin, g_end, std_functor(g_mean), 0.0, thrust::plus<float>()) / (N));
		b_std = sqrt(thrust::transform_reduce(b_begin, b_end, std_functor(b_mean), 0.0, thrust::plus<float>()) / (N));
	}
	else
	{
		r_std = sqrt(thrust::transform_reduce(r_begin, r_end, std_functor(0.0), 0.0, thrust::plus<float>()) / (N));
		g_std = sqrt(thrust::transform_reduce(g_begin, g_end, std_functor(0.0), 0.0, thrust::plus<float>()) / (N));
		b_std = sqrt(thrust::transform_reduce(b_begin, b_end, std_functor(0.0), 0.0, thrust::plus<float>()) / (N));
	}

	thrust::transform(r_begin, r_end, thrust::constant_iterator<float>(r_std), r_begin, thrust::divides<float>());
	thrust::transform(g_begin, g_end, thrust::constant_iterator<float>(g_std), g_begin, thrust::divides<float>());
	thrust::transform(b_begin, b_end, thrust::constant_iterator<float>(b_std), b_begin, thrust::divides<float>());

	if(out_r_std != NULL)
		*out_r_std = r_std;
	if(out_g_std != NULL)
		*out_g_std = g_std;
	if(out_b_std != NULL)
		*out_b_std = b_std;
}




float makeZeroMean(iu::TensorGpu_32f &d_inOut)
{
	float mean = cuda::mean(d_inOut);
	thrust::transform(d_inOut.begin(), d_inOut.end(), thrust::constant_iterator<float>(mean), d_inOut.begin(),
			thrust::minus<float>());

	return mean;
}

float makeUnitStd(iu::TensorGpu_32f &d_inOut, float mean)
{
	float std = cuda::std(d_inOut);
	thrust::transform(d_inOut.begin(), d_inOut.end(), thrust::constant_iterator<float>(std), d_inOut.begin(),
			thrust::divides<float>());

	return std;
}

float sum(iu::TensorGpu_32f &d_img)
{
	return thrust::reduce(d_img.begin(), d_img.end(), 0.0, thrust::plus<float>());
}

float mean(iu::TensorGpu_32f &d_img)
{
	int N = d_img.width() * d_img.height();
	return cuda::sum(d_img) / N;
}

float std(iu::TensorGpu_32f &d_img, float mean)
{
	int N = d_img.width() * d_img.height();

	if (mean == -11111111)
		mean = cuda::mean(d_img);

//	float eps = 1e-7;
//	if(mean > -eps && mean < eps)
//	{
//		return sqrt(thrust::transform_reduce(d_img.begin(), d_img.end(), square(), 0.0, thrust::plus<float>()) / (N - 1));
//	}
//	else
		return sqrt(
			thrust::transform_reduce(d_img.begin(), d_img.end(), std_functor(mean), 0.0, thrust::plus<float>())
					/ (N));
}

void negateTensor(iu::TensorGpu_32f &d_tensor)
{
	thrust::transform(d_tensor.begin(), d_tensor.end(), d_tensor.begin(), thrust::negate<float>());
}



void expTensor(iu::TensorGpu_32f &d_tensor, float alpha, float beta, float lambda)
{
	thrust::transform(d_tensor.begin(), d_tensor.end(), d_tensor.begin(), ExpEdges(alpha, beta, lambda));
}

void gradX(iu::TensorGpu_32f &d_inputImg, iu::TensorGpu_32f &d_gradX)
{
	dim3 threadsPerBlock(16,16);
	int H = d_inputImg.height();
	int W = d_inputImg.width();
	dim3 numBlocks(divup(W,threadsPerBlock.x),divup(H,threadsPerBlock.y));

	grad_x<<<numBlocks, threadsPerBlock>>>(d_gradX, d_inputImg);
	cudaDeviceSynchronize();
}

void gradY(iu::TensorGpu_32f &d_inputImg, iu::TensorGpu_32f &d_gradY)
{
	dim3 threadsPerBlock(16,16);
	int H = d_inputImg.height();
	int W = d_inputImg.width();
	dim3 numBlocks(divup(W,threadsPerBlock.x),divup(H,threadsPerBlock.y));

	grad_y<<<numBlocks, threadsPerBlock>>>(d_gradY, d_inputImg);
	cudaDeviceSynchronize();
}

__global__ void padding_color_kernel(iu::ImageGpu_32f_C4::KernelData output, cudaTextureObject_t input, int padval)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x < 0 || x >= output.width_ || y < 0 || y >= output.height_)
    	return;

    const float xx = (x - padval + 0.5f) / (output.width_ - 2 * padval);
	const float yy = (y - padval + 0.5f) / (output.height_ - 2 * padval);
	output(x, y) = tex2D<float4>(input, xx, yy);
}

void padColorImage(iu::ImageGpu_32f_C4 *out,iu::ImageGpu_32f_C4 *in, int padval)
{
    uint width = in->width()+2*padval;
    uint height = in->height()+2*padval;

    int gpu_block_x = 32;
    int gpu_block_y = 32;

    // compute number of Blocks
    int nb_x = divUp(width,gpu_block_x);
    int nb_y = divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    in->prepareTexture(cudaReadModeElementType, cudaFilterModeLinear, cudaAddressModeMirror);
    cudaTextureObject_t tex_in = in->getTexture();

    //    cudaTextureObject_t tex_in;
    //	bindTexture(tex_in,in,cudaAddressModeMirror);

    // perform padding
    padding_color_kernel<<<dimGrid,dimBlock,0>>>(*out, tex_in, padval);
    cudaDeviceSynchronize();
}

__global__ void kCropTensor(iu::TensorGpu_32f::TensorKernelData in, iu::TensorGpu_32f::TensorKernelData out, int crop)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z*blockDim.z + threadIdx.z;

    if(x < 0 || x >= out.W || y < 0 || y >= out.H)
    	return;

	out(0, c, y, x) = in(0, c, y + crop, x + crop);
}

void cropTensor(iu::TensorGpu_32f &d_in, iu::TensorGpu_32f &d_out, int crop)
{
	assert(d_in.width() == d_out.width() + 2*crop);
	assert(d_in.height() == d_out.height() + 2*crop);
	assert(d_in.memoryLayout() == d_out.memoryLayout());
	assert(d_in.memoryLayout() == iu::TensorGpu_32f::NCHW);

	dim3 threadsPerBlock(16,16,3);
	dim3 numBlocks(std::ceil((float)d_out.width() / threadsPerBlock.x),
			       std::ceil((float)d_out.height() / threadsPerBlock.y),
			       std::ceil((float)d_out.channels() / threadsPerBlock.z));

	kCropTensor<<<numBlocks, threadsPerBlock>>>(d_in, d_out, crop);
	cudaDeviceSynchronize();
}

void concatOver_c_dim(iu::TensorGpu_32f &first, iu::TensorGpu_32f &second, iu::TensorGpu_32f &output)
{
	assert(first.samples() == second.samples());
	assert(first.width() == second.width());
	assert(first.height() == second.height());
	assert(first.height() == output.height());
	assert(first.width() == output.width());
    assert(first.samples() == output.samples());
	assert(first.channels() + second.channels() == output.channels());

	dim3 threadsPerBlock(16, 16, 1);
	dim3 numBlocks(std::ceil((float)output.width() / threadsPerBlock.x),
				   std::ceil((float)output.height() / threadsPerBlock.y),
				   std::ceil((float)output.channels() / threadsPerBlock.z));

	kConncatChannels<<<numBlocks, threadsPerBlock>>>(first, second, output);
	cudaDeviceSynchronize();

}

}
