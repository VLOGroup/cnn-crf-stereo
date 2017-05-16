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

#ifndef EXAMPLE_CU
#define EXAMPLE_CU

#include "stereo.cuh"
#include "iu/iucore.h"
#include "iu/iuio.h"
#include "opencv2/highgui/highgui.hpp"
#include "iu/iuhelpermath.h"
#include "colorstereonet.h"

// defines
#define GPU_BLOCK_SIZE 32
#define FILTER_SIZE_MAX 5

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

//#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
//#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

//inline void __cudaSafeCall( cudaError err, const char *file, const int line )
//{
//#ifdef CUDA_ERROR_CHECK
//if ( cudaSuccess != err )
//{
//fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
//file, line, cudaGetErrorString( err ) );
//exit( -1 );
//}
//#endif

//return;
//}

//inline void __cudaCheckError( const char *file, const int line )
//{
//#ifdef CUDA_ERROR_CHECK
//cudaError err = cudaGetLastError();
//if ( cudaSuccess != err )
//{
//fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
//file, line, cudaGetErrorString( err ) );
//exit( -1 );

//}
//#endif

//}

// ugly global variables
static cudaTextureObject_t tex_u, tex_u_, tex_p, tex_q;

//static GrayStereoNet* stereoNet = NULL;
static iu::TensorGpu_32f *costvolume_refinement_ = NULL;
//static iu::ImageGpu_32f_C1 *I1_padded = NULL;
//static iu::ImageGpu_32f_C1 *I2_padded = NULL;
//static int padval;

extern void save(iu::TensorGpu_32f &tensor, std::string path);

//cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

//texture<float, 2, cudaReadModeElementType> tex_img2;
//__constant__ float3 const_fundamental[3];

////////////////////////////////////////////////////////////////////////////////
void bindTexture(cudaTextureObject_t& tex, iu::ImageGpu_32f_C1* mem, cudaTextureAddressMode addr_mode, cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = mem->data();
    resDesc.res.pitch2D.pitchInBytes = mem->pitch();
    resDesc.res.pitch2D.width = mem->width();
    resDesc.res.pitch2D.height = mem->height();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = (addr_mode==cudaAddressModeMirror);
    texDesc.addressMode[0] = addr_mode;
    texDesc.addressMode[1] = addr_mode;
    texDesc.filterMode = filter_mode;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
}

////////////////////////////////////////////////////////////////////////////////
void bindTexture(cudaTextureObject_t& tex, iu::ImageGpu_32f_C2* mem, cudaTextureAddressMode addr_mode, cudaTextureFilterMode filter_mode=cudaFilterModeLinear)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = mem->data();
    resDesc.res.pitch2D.pitchInBytes = mem->pitch();
    resDesc.res.pitch2D.width = mem->width();
    resDesc.res.pitch2D.height = mem->height();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float2>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = (addr_mode==cudaAddressModeMirror);
    texDesc.addressMode[0] = addr_mode;
    texDesc.addressMode[1] = addr_mode;
    texDesc.filterMode = filter_mode;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
}

////////////////////////////////////////////////////////////////////////////////
void bindTexture(cudaTextureObject_t& tex, iu::ImageGpu_32f_C4* mem, cudaTextureAddressMode addr_mode)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = mem->data();
    resDesc.res.pitch2D.pitchInBytes = mem->pitch();
    resDesc.res.pitch2D.width = mem->width();
    resDesc.res.pitch2D.height = mem->height();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float4>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = (addr_mode==cudaAddressModeMirror);
    texDesc.addressMode[0] = addr_mode;
    texDesc.addressMode[1] = addr_mode;
    texDesc.filterMode = cudaFilterModePoint;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
}

////////////////////////////////////////////////////////////////////////////////
void bindTexture(cudaTextureObject_t& tex, iu::VolumeGpu_32f_C1* mem, int slice, cudaTextureAddressMode addr_mode)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = mem->data(0,0,slice);
    resDesc.res.pitch2D.pitchInBytes = mem->pitch();
    resDesc.res.pitch2D.width = mem->width();
    resDesc.res.pitch2D.height = mem->height();
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = (addr_mode==cudaAddressModeMirror);
    texDesc.addressMode[0] = addr_mode;
    texDesc.addressMode[1] = addr_mode;
    texDesc.filterMode = cudaFilterModeLinear;

    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
}

struct floatint{
public:
    float val;
    int ind;
public:
    __host__ __device__ floatint() {
        val = 1e5;
        ind = -1;
    };

    __host__ __device__ floatint(float val_, int ind_){
        val = val_;
        ind = ind_;
    };

    __host__ __device__ floatint(const floatint& x) {
        this->set(x.val, x.ind);
    }
    __host__ __device__ floatint(const volatile floatint& x) {
        this->set(x.val, x.ind);
    }

    __host__ __device__ void set(float val_, int ind_) volatile {
        val = val_;
        ind = ind_;
    };

    __host__ __device__ volatile floatint & operator += (volatile floatint & b) volatile {
        if(b.val < val){
            //this->set(b.val, b.ind);
            val = b.val;
            ind = b.ind;
        };
        return *this;
    };
};


__device__ floatint localMin(float val, volatile floatint* reducionSpace, uint localId, uint num)
{

    if(localId < num){
        reducionSpace[localId].set(val,localId);  // load data into shared mem
    };
    __syncthreads();

    //  for(uint next = (num+1)/2; next > 32; next = (next+1)/2)
    //  {
    //    __syncthreads();
    //    if(localId < next)
    //      reducionSpace[localId] = reducionSpace[localId] + reducionSpace[next + localId];
    //  }

    // complete loop unroll
    if( num > 128){
        if (localId < num - 128) {
            reducionSpace[localId] += reducionSpace[localId + 128];
        };
        __syncthreads();
    };

    if( num > 64){
        if (localId< num - 64) {
            reducionSpace[localId] += reducionSpace[localId + 64];
        };
        __syncthreads();
    };

    // within one warp (=32 threads) instructions are SIMD synchronous
    // -> __syncthreads() not needed
    if( num > 32){
        if (localId < 32){
            reducionSpace[localId] += reducionSpace[localId+32];
        };
    };
    if (localId < 16)
    {
        reducionSpace[localId] += reducionSpace[localId+16];
        reducionSpace[localId] += reducionSpace[localId+8];
        reducionSpace[localId] += reducionSpace[localId+4];
        reducionSpace[localId] += reducionSpace[localId+2];
        reducionSpace[localId] += reducionSpace[localId+1];
    }

    __syncthreads();

    return reducionSpace[0];
}

inline __device__ float sum(float3 val)
{
    return val.x+val.y+val.z;
}
inline __device__ float sum(float2 val)
{
    return val.x+val.y;
}

inline __device__ float2 abs(float2 val)
{
    return make_float2(abs(val.x),abs(val.y));
}

template<class T>
__global__ void set_value_kernel(T value, T* dst, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<width && y<height)
    {
        dst[y*stride+x] = value;
    }
}

__inline__ __device__ char getTernaryCensus(float I1, float I2, float eps)
{
	float diff = I2-I1;
	if((abs(diff)-eps)<0)
		return 2;
    if((abs(diff)+eps)<0)
		return 0;
	else
		return 1;
}

__inline__ __device__ float softCensus(float p, float c)
{
    return (c-p)/max(0.01f,abs(c-p));
}

//////////////////////////////////////////////////////////////////////////////////
__global__ void cost_vol_census_kernel(iu::LinearDeviceMemory_32f_C1::KernelData volume,
								cudaTextureObject_t tex_I1, cudaTextureObject_t tex_I2,
                                float disp_min, float disp_step, int filter_width, int filter_height, float lambda,
                                int width, int height, int depth)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float xx = x + 0.5f;
    const float yy = y + 0.5f;

    if(x<width && y<height)
    {
        float disp = disp_min;
        float I1_c = tex2D<float>(tex_I1,xx, yy);
        // loop over disparities
        for(int z=0;z<depth;z++)
        {
            float it_tmp = 0.f;
            float I2_c = tex2D<float>(tex_I2,xx - disp, yy);
            for(float dy=yy-filter_height;dy<=yy+filter_height;dy++)
            {
                for(float dx=xx-filter_width;dx<=xx+filter_width;dx++)
                {
                    if(dx==xx && dy==yy)
                        continue;
                    float I1 = tex2D<float>(tex_I1,dx, dy);
                    float I2 = tex2D<float>(tex_I2,dx-disp, dy);

                    if((I1_c>I1) != (I2_c>I2))
                        ++it_tmp;
//                    it_tmp += abs(softCensus(I1,I1_c)-softCensus(I2,I2_c));
                }

            }
            volume.data_[y*width*depth + x*depth + z] = lambda*((it_tmp/(filter_height*filter_width*4)));
            disp+=disp_step;
        }
    }

}

__global__ void arg_min_kernel(iu::ImageGpu_32f_C1::KernelData out, iu::LinearDeviceMemory_32f_C1::KernelData volume, int depth, float disp_min, float disp_step)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;


    if(x<out.width_ && y<out.height_)
    {
        float cost_min = FLT_MAX;
        int min_idx = 0;
        // loop over disparities
        for(int z=0;z<depth;z++)
        {
            float curr_cost = volume.data_[y*out.width_*depth + x*depth + z];
            if(curr_cost<cost_min) {
                cost_min = curr_cost;
                min_idx = z;
            }
        }
        out(x,y) = disp_min + min_idx*disp_step;
    }

}

__global__ void cost_vol_census_min_kernel(iu::LinearDeviceMemory_32f_C1::KernelData volume,
                                cudaTextureObject_t tex_I1, cudaTextureObject_t tex_I2,
                                float disp_min, float disp_step, int filter_width, int filter_height, float lambda,
                                int width, int height, int depth)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float xx = x + 0.5f;
    const float yy = y + 0.5f;

    if(x<width && y<height)
    {
        float disp = disp_min;
        float I1_c = tex2D<float>(tex_I1,xx, yy);
        // loop over disparities
        for(int z=0;z<depth;z++)
        {
            float it = filter_width*filter_height*4;
            for(int ddy=-3;ddy<=3;ddy++)
            {
                float it_tmp = 0.f;
                float I2_c = tex2D<float>(tex_I2,xx - disp, yy+ddy);
                for(float dy=yy-filter_height;dy<=yy+filter_height;dy++)
                {
                    for(float dx=xx-filter_width;dx<=xx+filter_width;dx++)
                    {
                        if(dx==xx && dy==yy)
                            continue;
                        float I1 = tex2D<float>(tex_I1,dx, dy);
                        float I2 = tex2D<float>(tex_I2,dx-disp, dy+ddy);

                        if((I1_c>I1) != (I2_c>I2))
                            ++it_tmp;
    //                    it_tmp += abs(softCensus(I1,I1_c)-softCensus(I2,I2_c));
                    }

                }
                if(it_tmp<it)
                    it=it_tmp;
            }
            volume.data_[y*width*depth + x*depth + z] = lambda*((it/(filter_height*filter_width*4)));
            disp+=disp_step;
        }
    }
}

__global__ void do_fitting_kernel(iu::ImageGpu_32f_C2::KernelData device_out, iu::TensorGpu_32f::TensorKernelData volume, float disp_step, float lambda, bool do_abs)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        // volume is a width x height x 3 cost volume (x+-h)
        float2 out;
//        const int center = y*device_out.width_*3 + x*3 + 1;
        const float c[3]={volume(0,y,x,0),volume(0,y,x,1),volume(0,y,x,2)};
        const float h = 0.25*disp_step;
        if(do_abs) {
            out.x = (c[2]-c[1])/h*lambda;
            out.y = (c[1]-c[0])/h*lambda;
            if(out.y>out.x)
                out = make_float2(out.x*0.5f + out.y*0.5f);
        } else {
            out.x = (c[0]-c[2])/(h*2.f)*lambda;
            out.y = max(0.f,(c[0]+c[2]-2*c[1])/(h*h)*lambda);
        }
        device_out(x,y) = out;
    }
}

__global__ void mod_vol_census_kernel(iu::VolumeGpu_32f_C1::KernelData vol,cudaTextureObject_t tex_x,cudaTextureObject_t tex_w,
		float C, float disp_min, float disp_step, bool horizontal,iu::VolumeGpu_32f_C1::KernelData vol_out)
{

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	const float xx = x + 0.5f;
	const float yy = y + 0.5f;

	if(x<vol.height_ && y<vol.depth_)
	{
		// read values around current x|y location
		float x_lu = (tex2D<float>(tex_x,horizontal?xx:xx-1,horizontal?yy-1:yy)-disp_min)/disp_step;
		float x_rd = (tex2D<float>(tex_x,horizontal?xx:xx+1,horizontal?yy+1:yy)-disp_min)/disp_step;
		float w_lu = tex2D<float>(tex_w,horizontal?xx:xx-1,horizontal?yy-1:yy);
		float w_rd = tex2D<float>(tex_w,horizontal?xx:xx+1,horizontal?yy+1:yy);


		for(unsigned int z=0;z<vol.width_;z++)
		{
            float cost_update = min(C,abs(x_lu-z))*w_lu + min(C,abs(x_rd-z))*w_rd;
			vol_out.data_[y*vol.stride_*vol.height_ + x*vol.stride_ + z] = vol.data_[y*vol.stride_*vol.height_ + x*vol.stride_ + z] + cost_update;
		}
	}
}

__global__ void filter_x_forward_kernel(iu::ImageGpu_32f_C1::KernelData device_out, cudaTextureObject_t tex_in, float alpha, float beta, float lambda)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        // Define arrays for shared memory
        __shared__ float input_shared[GPU_BLOCK_SIZE][GPU_BLOCK_SIZE+1];
        // load data into shared memory
        input_shared[ty][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        // border threads have more work to do
        if (tx == GPU_BLOCK_SIZE-1)
            input_shared[ty][tx+1] = tex2D<float>(tex_in,(x+1.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        //if (tx == 0)
        //    input_shared[ty][tx-1] = tex2D<float>(tex_in,x-0.5f,y+0.5f);
        __syncthreads();

        // calculate the forward difference
        float diff = abs(input_shared[ty][tx+1]-input_shared[ty][tx]);
        device_out.data_[y*device_out.stride_+x] = (alpha>0)?max(lambda*exp(-alpha*pow(diff,beta)),1.e-6f):lambda*diff;
    }
}

__global__ void filter_x_backward_kernel(iu::ImageGpu_32f_C1::KernelData device_out, cudaTextureObject_t tex_in, float alpha, float beta, float lambda)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        // Thread index
        int tx = threadIdx.x+1;
        int ty = threadIdx.y;
        // Define arrays for shared memory
        __shared__ float input_shared[GPU_BLOCK_SIZE][GPU_BLOCK_SIZE+1];
        // load data into shared memory
        input_shared[ty][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        // border threads have more work to do
//        if (tx == GPU_BLOCK_SIZE-1)
//            input_shared[ty][tx+1] = tex2D<float>(tex_in,(x+1.5f)/width,(y+0.5f)/height);
        if (tx == 1)
            input_shared[ty][tx-1] = tex2D<float>(tex_in,(x-0.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        __syncthreads();

        // calculate the backward difference
        float diff = -input_shared[ty][tx-1]+input_shared[ty][tx];
        device_out.data_[y*device_out.stride_+x] = (alpha>0)?max(lambda*exp(-alpha*pow(diff,beta)),1.e-6f):lambda*diff;
    }
}

__global__ void filter_y_forward_kernel(iu::ImageGpu_32f_C1::KernelData device_out, cudaTextureObject_t tex_in, float alpha, float beta, float lambda)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        // Define arrays for shared memory
        __shared__ float input_shared[GPU_BLOCK_SIZE+1][GPU_BLOCK_SIZE];
        // load data into shared memory
        input_shared[ty][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        // border threads have more work to do
        if (ty == GPU_BLOCK_SIZE-1)
            input_shared[ty+1][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y+1.5f)/device_out.height_);
        //if (tx == 0)
        //    input_shared[ty][tx-1] = tex2D<float>(tex_in,x-0.5f,y+0.5f);
        __syncthreads();

        // calculate the forward difference
        float diff = abs(input_shared[ty+1][tx]-input_shared[ty][tx]);
        device_out.data_[y*device_out.stride_+x] = (alpha>0)?max(lambda*exp(-alpha*pow(diff,beta)),1.e-6f):lambda*diff;
    }
}

__global__ void filter_y_backward_kernel(iu::ImageGpu_32f_C1::KernelData device_out, cudaTextureObject_t tex_in, float alpha, float beta, float lambda)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y+1;
        // Define arrays for shared memory
        __shared__ float input_shared[GPU_BLOCK_SIZE+1][GPU_BLOCK_SIZE];
        // load data into shared memory
        input_shared[ty][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y+0.5f)/device_out.height_);
        // border threads have more work to do
//        if (tx == GPU_BLOCK_SIZE-1)
//            input_shared[ty][tx+1] = tex2D<float>(tex_in,(x+1.5f)/width,(y+0.5f)/height);
        if (ty == 1)
            input_shared[ty-1][tx] = tex2D<float>(tex_in,(x+0.5f)/device_out.width_,(y-0.5f)/device_out.height_);
        __syncthreads();

        // calculate the backward difference
        float diff = -input_shared[ty-1][tx]+input_shared[ty][tx];
        device_out.data_[y*device_out.stride_+x] = (alpha>0)?max(lambda*exp(-alpha*pow(diff,beta)),1.e-6f):lambda*diff;
    }
}

inline __device__ float4 make_float4(float2 val1,float2 val2)
{
    return make_float4(val1.x,val1.y,val2.x,val2.y);
}

__global__ void tvqf_warp_live_occ_kernel (iu::ImageGpu_32f_C1::KernelData u,iu::ImageGpu_8u_C1::KernelData occlusion_mask, cudaTextureObject_t tex_I1, cudaTextureObject_t tex_I2,
                                           int filter_size, float lambda,
                                           iu::ImageGpu_32f_C2::KernelData qf,float disp_step, bool do_abs)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float xx = x + 0.5f;
    const float yy = y + 0.5f;
    const float h=disp_step*0.25f;
    if(x<u.width_ && y<u.height_)
    {
       float2 out = make_float2(0.f);
       if(occlusion_mask(x,y)!=1) { // if there is an occlusion, simply set the data term to 0
           const float disp = u(x,y);
           // loop over disparities
           float c[3]={0.f,0.f,0.f};
           float I2_c[3];
           I2_c[0] = tex2D<float>(tex_I2,xx - disp - h, yy);
           I2_c[1] = tex2D<float>(tex_I2,xx - disp    , yy);
           I2_c[2] = tex2D<float>(tex_I2,xx - disp + h, yy);
           float I2;
           float I1;
           float I1_c = tex2D<float>(tex_I1,xx, yy);
           for(float dy=yy-filter_size;dy<=yy+filter_size;dy++)
           {
               for(float dx=xx-filter_size;dx<=xx+filter_size;dx++)
               {
                   if(dx==xx && dy==yy)
                       continue;
                   I1 = tex2D<float>(tex_I1,dx, dy);
                   bool c1 = I1_c>I1;
                   // calculate all 3 censuses at once
                   I2 = tex2D<float>(tex_I2,dx-disp - h, dy);
                   if(c1 != (I2_c[0]>I2))
                       ++c[0];
                   I2 = tex2D<float>(tex_I2,dx-disp    , dy);
                   if(c1 != (I2_c[1]>I2))
                       ++c[1];
                   I2 = tex2D<float>(tex_I2,dx-disp + h, dy);
                   if(c1 != (I2_c[2]>I2))
                       ++c[2];
               }
           }
           float normalizer = lambda/(filter_size*filter_size*4);
           if(do_abs) {
               out.x = (c[2]-c[1])/h*normalizer;
               out.y = (c[1]-c[0])/h*normalizer;
               if(out.y>out.x)
                   out = make_float2(out.x*0.5f + out.y*0.5f);
           } else {
               out.x = (c[0]-c[2])/(h*2.f)*normalizer;
               out.y = max(0.f,(c[0]+c[2]-2*c[1])/(h*h)*normalizer);
           }
       }
       qf(x,y) = out;
    }
}

 __global__ void tvqf_warp_live_kernel (iu::ImageGpu_32f_C1::KernelData u,cudaTextureObject_t tex_I1, cudaTextureObject_t tex_I2,
                                        int filter_size, float lambda,
                                        iu::ImageGpu_32f_C2::KernelData qf,float disp_step, bool do_abs)
 {
     const int x = blockIdx.x*blockDim.x + threadIdx.x;
     const int y = blockIdx.y*blockDim.y + threadIdx.y;

     const float xx = x + 0.5f;
     const float yy = y + 0.5f;
     const float h=disp_step*0.25f;
     if(x<u.width_ && y<u.height_)
     {
        const float disp = u(x,y);
        float2 out = make_float2(0.f);

        // loop over disparities
        float c[3]={0.f,0.f,0.f};
        float I2_c[3];
        I2_c[0] = tex2D<float>(tex_I2,xx - disp - h, yy);
        I2_c[1] = tex2D<float>(tex_I2,xx - disp    , yy);
        I2_c[2] = tex2D<float>(tex_I2,xx - disp + h, yy);
        float I2;
        float I1;
        float I1_c = tex2D<float>(tex_I1,xx, yy);
        for(float dy=yy-filter_size;dy<=yy+filter_size;dy++)
        {
            for(float dx=xx-filter_size;dx<=xx+filter_size;dx++)
            {
                if(dx==xx && dy==yy)
                    continue;
                I1 = tex2D<float>(tex_I1,dx, dy);
                bool c1 = I1_c>I1;
                // calculate all 3 censuses at once
                I2 = tex2D<float>(tex_I2,dx-disp - h, dy);
                if(c1 != (I2_c[0]>I2))
                    ++c[0];
                I2 = tex2D<float>(tex_I2,dx-disp    , dy);
                if(c1 != (I2_c[1]>I2))
                    ++c[1];
                I2 = tex2D<float>(tex_I2,dx-disp + h, dy);
                if(c1 != (I2_c[2]>I2))
                    ++c[2];
            }
        }
        float normalizer = lambda/(filter_size*filter_size*4);
        if(do_abs) {
            out.x = (c[2]-c[1])/h*normalizer;
            out.y = (c[1]-c[0])/h*normalizer;
            if(out.y>out.x)
                out = make_float2(out.x*0.5f + out.y*0.5f);
        } else {
            out.x = (c[0]-c[2])/(h*2.f)*normalizer;
            out.y = max(0.f,(c[0]+c[2]-2*c[1])/(h*h)*normalizer);
        }
        qf(x,y) = out;
     }
 }

  __global__ void ttvqf_primal_kernel(iu::ImageGpu_32f_C1::KernelData u,iu::ImageGpu_32f_C1::KernelData u_,iu::ImageGpu_32f_C1::KernelData u_0,
                                    iu::ImageGpu_32f_C2::KernelData qf,cudaTextureObject_t tex_p,cudaTextureObject_t tex_q,
                                    float tau,float disp_step)
 {
     const int x = blockIdx.x*blockDim.x + threadIdx.x;
     const int y = blockIdx.y*blockDim.y + threadIdx.y;

     if (x<u.width_ && y<u.height_)
     {
         // texture coordinates
         const int c = y*u.stride_+x;
         const float xx = (x + 0.5f);
         const float yy = (y + 0.5f);
         float2 P_xy = tex2D<float2>(tex_p,xx,yy)-tex2D<float2>(tex_q,xx,yy);
         float divergence = P_xy.x - (tex2D<float2>(tex_p,xx-1.f,yy).x-tex2D<float2>(tex_q,xx-1.f,yy).x) +
                            P_xy.y - (tex2D<float2>(tex_p,xx,yy-1.f).y-tex2D<float2>(tex_q,xx,yy-1.f).y);
         // update
         float u_new = u.data_[c] + tau * divergence;
         float2 ab=qf(x,y);

         // projection
         float u0=u_0(x,y);
         u_new = clamp((u_new-(tau*(ab.x-ab.y*u0)))/(1+tau*ab.y),u0-0.5*disp_step,u0+0.5*disp_step);

         // over-relaxation
         u_.data_[c] = 2 * u_new - u_.data_[c];
         u.data_[c] = u_new;
     }
 }

 
 __device__ inline float2 sign(float2 val)
 {
     return make_float2(val.x>0?1.f:-1.f,val.y>0?1.f:-1.f);
 }

 __device__ inline float sign(float val)
 {
     return val>0?1.f:-1.f;
 }

 __device__ inline float2 relu(float2 val)
 {
     return make_float2(val.x>0?val.x:0,val.y>0?val.y:0 );
 }

 __device__ inline float2 max(float2 val1,float2 val2)
 {
     return make_float2(max(val1.x,val2.x),max(val1.y,val2.y));
 }

 __device__ inline float prox_p(float p_new, float alpha, float beta, float sigma, float w)
 {
     if(abs(p_new)<=w* alpha)
        return p_new;
     return clamp(max(alpha*w,abs(p_new)-sigma*beta)*sign(p_new),-w,w);
 }

 __device__ inline float2 prox_p(float2 p_new, float alpha, float beta, float sigma, float2 w)
 {
     return make_float2(prox_p(p_new.x,alpha,beta,sigma,w.x),
                        prox_p(p_new.y,alpha,beta,sigma,w.y));
 }

 __device__ inline float4 prox_p(float4 p_new, float alpha, float beta, float sigma, float2 w)
 {
     return make_float4(prox_p(p_new.x,alpha,beta,sigma,w.x),
                        prox_p(p_new.y,alpha,beta,sigma,w.x),
                        prox_p(p_new.z,alpha,beta,sigma,w.y),
						prox_p(p_new.w,alpha,beta,sigma,w.y));
 }

 __global__ void thuberqf_dual_kernel(iu::ImageGpu_32f_C2::KernelData p, iu::ImageGpu_32f_C2::KernelData q,
                                   iu::ImageGpu_32f_C1::KernelData wx,iu::ImageGpu_32f_C1::KernelData wy,
                                   cudaTextureObject_t tex_u,float sigma,float C)
 {
     const int x = blockIdx.x*blockDim.x + threadIdx.x;
     const int y = blockIdx.y*blockDim.y + threadIdx.y;

     if (x<p.width_ && y<p.height_)
     {
         // texture coordinates
         const float xx = (x + 0.5f);
         const float yy = (y + 0.5f);
         float u_xy = tex2D<float>(tex_u,xx,yy);
         float2 du = make_float2(
                     tex2D<float>(tex_u,xx+1.f,yy)-u_xy,
                     tex2D<float>(tex_u,xx,yy+1.f)-u_xy);
         float2 p_new = p(x,y) + sigma*du;
         float2 w = make_float2(wx(x,y),wy(x,y));
//         p(x,y) = prox_p(p_new,0.5f,1.f,sigma,w);
         p(x,y) = clamp(p_new/(1+sigma*0.1f),-w,w);
         float2 q_new = q(x,y) + sigma*du;
         q(x,y) = prox_p(q_new,0.f,C,sigma,w);
     }
 }

#undef DISPARITIES
#undef IMG_WIDTH
#undef IMG_HEIGHT
#undef STRIDE

__global__ void leftrightcheck_kernel(iu::ImageGpu_32f_C1::KernelData device_out, cudaTextureObject_t tex_left, cudaTextureObject_t tex_right, float th)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    const float xx = x + 0.5f;
    const float yy = y + 0.5f;

    if(x<device_out.width_ && y<device_out.height_)
    {
        float I1 = tex2D<float>(tex_left,xx,yy);
        float I2_warped = -tex2D<float>(tex_right,xx-I1,yy);
        device_out.data_[y*device_out.stride_+x]=abs(I1-I2_warped)<=th?I1:-1.0f;
    }
}

__global__ void warp_image_kernel(iu::ImageGpu_32f_C1::KernelData device_out, iu::ImageGpu_32f_C1::KernelData disparity, cudaTextureObject_t tex_input)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        device_out(x,y) = tex2D<float>(tex_input,x+0.5f-disparity(x,y),y+0.5f);
    }
}

__global__ void warp_image_kernel(iu::ImageGpu_32f_C4::KernelData device_out, iu::ImageGpu_32f_C1::KernelData disparity, cudaTextureObject_t tex_input)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        device_out(x,y) = tex2D<float4>(tex_input,x+0.5f-disparity(x,y),y+0.5f);
    }
}

__global__ void remapsolution_kernel(iu::ImageGpu_32f_C1::KernelData device_out, iu::ImageGpu_32s_C1::KernelData device_in, float disp_min, float disp_step)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_out.width_ && y<device_out.height_)
    {
        device_out(x,y) = ((float)device_in(x,y)*disp_step)+disp_min;
    }
}


__global__ void rgb_to_gray_kernel(iu::ImageGpu_32f_C4::KernelData rgb, iu::ImageGpu_32f_C1::KernelData gray)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<rgb.width_&& y<rgb.height_)
    {
        float4 rgb_val = rgb(x,y);
        gray(x,y) = 0.2989 * rgb_val.x + 0.5870 * rgb_val.y + 0.1140 * rgb_val.z;
    }
}

__global__ void padding_kernel(iu::ImageGpu_32f_C1::KernelData output, cudaTextureObject_t input, int padval)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x<output.width_&& y<output.height_)
    {
        const float xx = (x-padval+0.5f)/(output.width_-2*padval);
        const float yy = (y-padval+0.5f)/(output.height_-2*padval);
        output(x,y) = tex2D<float>(input,xx,yy);
    }
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

//    // center
//    if(x >= padval && y >= padval && x < output.width_- padval && y < output.height_ - padval)
//    	output(x, y) = input(x - padval, y - padval);
//    else if(x < padval && y < padval) // top-left
//    	output(x, y) = input(0, 0);
//    else if(x >=  output.width_- padval && y < padval) // top-right
//    	output(x, y) = input(input.width_ - 1, 0);
//    else if(x >=  output.width_- padval && y >=  output.height_- padval) // bottom-right
//    	output(x, y) = input(input.width_ - 1, input.height_ - 1);
//    else if(x < padval && y >= output.height_ - padval) // bottom -left
//    	output(x, y) = input(0, input.height_ - 1);
//    else if(y < padval) // top
//    	output(x, y) = input(x, abs(y - padval) - 1);
//    else if(x >= output.width_ - padval) // right
//    	output(x, y) = input(padval + input.width_ - x + input.width_ - 1, y);
//    else if(y >= output.height_ - padval) // bottom
//    	output(x, y) = input(x, padval + input.height_ - y + input.height_ - 1);
//    else if(x < padval) // left
//    	output(x, y) = input(abs(x - padval) - 1, y);

}

__global__ void do_fitting_kernel_volume(iu::ImageGpu_32f_C2::KernelData device_out, iu::ImageGpu_32f_C1::KernelData slack_prop_out, kernel::ndarray_ref<float,3> volume, float disp_min, float disp_step, float lambda){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if( x < volume.size(0) && y < volume.size(1)){
        // volume is a width x height x depth cost volume
        float2 out;
        int z = (slack_prop_out(x,y)-disp_min)/disp_step;
        if( z==0 || z==volume.size(2)) {
            out = make_float2(0.f,0.f);
        } else {
            const float c[3]={volume(x,y,z-1),volume(x,y,z),volume(x,y,z+1)};
            const float h = disp_step;
            out.x = (c[0]-c[2])/(h*2.f)*lambda;
            out.y = max(1e-3f,(c[0]+c[2]-2*c[1])/(h*h)*lambda);
        }
        device_out(x,y) = out;
    }
}

__global__ void do_argmin_kernel_volume(iu::ImageGpu_32f_C1::KernelData device_inout,  iu::ImageGpu_32f_C2::KernelData qf){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<device_inout.width_ && y<device_inout.height_) {
        float2 qfc = qf(x,y);
        if(qfc.y!=0)
            device_inout(x,y) += qfc.x/qfc.y;
    }
}


namespace cuda {
inline uint divUp(uint a, uint b) { return (a + b - 1) / b; }

void calccostvolumecensus(iu::ImageGpu_32f_C1* I1, iu::ImageGpu_32f_C1* I2,
                    float lambda, int filter_size, float disp_min, float disp_step, float disp_max,
                    iu::LinearDeviceMemory_32f_C1* costvolume, cudaStream_t stream)
{
    uint width = I1->width();
    uint height = I1->height();
    uint steps = costvolume->numel()/(width*height);

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = divUp(width,gpu_block_x);
    int nb_y = divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    cudaTextureObject_t tex_I1, tex_I2;
    bindTexture(tex_I1,I1,cudaAddressModeClamp);
    bindTexture(tex_I2,I2,cudaAddressModeClamp);

    cost_vol_census_kernel<<<dimGrid,dimBlock,0,stream>>>(*costvolume,tex_I1,tex_I2,disp_min,disp_step,
                    filter_size,filter_size,lambda,width,height,steps);
    CudaCheckError();
    cudaDestroyTextureObject(tex_I1);
    cudaDestroyTextureObject(tex_I2);
}

void computeArgmin(iu::ImageGpu_32f_C1* device_out, iu::LinearDeviceMemory_32f_C1* costvolume, float disp_min, float disp_step)
{
    uint width = device_out->width();
    uint height = device_out->height();
    uint steps = costvolume->numel()/(width*height);

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = divUp(width,gpu_block_x);
    int nb_y = divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    arg_min_kernel<<<dimGrid,dimBlock,0>>>(*device_out,*costvolume,steps, disp_min, disp_step);
    CudaCheckError();
}

void padImage(iu::ImageGpu_32f_C1 *out,iu::ImageGpu_32f_C1 *in, int padval)
{
    uint width = in->width()+2*padval;
    uint height = in->height()+2*padval;

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = divUp(width,gpu_block_x);
    int nb_y = divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    cudaTextureObject_t tex_in;
    bindTexture(tex_in,in,cudaAddressModeMirror);

    // perform padding
    padding_kernel<<<dimGrid,dimBlock,0>>>(*out,tex_in,padval);
    CudaCheckError();
    cudaDestroyTextureObject(tex_in);
    cudaDeviceSynchronize();
}

void padColorImage(iu::ImageGpu_32f_C4 *out,iu::ImageGpu_32f_C4 *in, int padval)
{
    uint width = in->width()+2*padval;
    uint height = in->height()+2*padval;

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = divUp(width,gpu_block_x);
    int nb_y = divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    cudaTextureObject_t tex_in;
	bindTexture(tex_in,in,cudaAddressModeMirror);

    // perform padding
    padding_color_kernel<<<dimGrid,dimBlock,0>>>(*out, tex_in, padval);
    CudaCheckError();
    cudaDeviceSynchronize();
}

iu::TensorGpu_32f *calccostvolumeColorCNN(iu::ImageGpu_32f_C4 *I1, iu::ImageGpu_32f_C4 *I1_padded, iu::ImageGpu_32f_C4 *I2, iu::ImageGpu_32f_C4 *I2_padded,
		ColorStereoNet *stereoNet, int padval)
{
	padColorImage(I1_padded, I1, padval);
    padColorImage(I2_padded, I2, padval);
    if(!costvolume_refinement_)
        costvolume_refinement_ = new iu::TensorGpu_32f(1,3,I1->height(),I1->width(),iu::TensorGpu_32f::MemoryLayout::NHWC);

    iu::TensorGpu_32f* d_out = stereoNet->predict(I1_padded, I2_padded);
    return d_out;
}

void filter(iu::ImageGpu_32f_C1* device_in,iu::ImageGpu_32f_C1* device_out, bool x, bool forward, float alpha, float beta, float lambda)
{
    uint width = device_out->width();
    uint height = device_out->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    cudaTextureObject_t tex_in;
    bindTexture(tex_in, device_in, cudaAddressModeMirror);

    if(x)
        if(forward)
            filter_x_forward_kernel<<<dimGrid,dimBlock,0,0>>>(*device_out, tex_in, alpha, beta,lambda);
        else
            filter_x_backward_kernel<<<dimGrid,dimBlock,0,0>>>(*device_out, tex_in, alpha, beta,lambda);
    else
        if(forward)
            filter_y_forward_kernel<<<dimGrid,dimBlock,0,0>>>(*device_out, tex_in, alpha, beta, lambda);
        else
            filter_y_backward_kernel<<<dimGrid,dimBlock,0,0>>>(*device_out, tex_in, alpha, beta, lambda);
    cudaDestroyTextureObject(tex_in);

}

void calcEdgeImage(iu::ImageGpu_32f_C1* I1,iu::ImageGpu_32f_C1* temp, iu::ImageGpu_32f_C1* wx, iu::ImageGpu_32f_C1* wy, float alpha, float beta, float lambda)
{
    iu::filterGauss(I1,temp,0.7);
    filter(temp,wx,true,true,alpha, beta, lambda);
    filter(temp,wy,false,true,alpha, beta, lambda);
}

void prepareFuseTTVL1(iu::ImageGpu_32f_C1* u,iu::ImageGpu_32f_C1* u_ ,iu::ImageGpu_32f_C2* p,iu::ImageGpu_32f_C2* q)
{
    bindTexture(tex_u_,u_,cudaAddressModeClamp,cudaFilterModePoint);
    bindTexture(tex_u,u,cudaAddressModeClamp,cudaFilterModePoint);
    bindTexture(tex_p,p,cudaAddressModeClamp);;
    bindTexture(tex_q,q,cudaAddressModeClamp);
}

void unprepareFuseTTVL1()
{
    cudaDestroyTextureObject(tex_u_);
    cudaDestroyTextureObject(tex_u);
    cudaDestroyTextureObject(tex_p);
    cudaDestroyTextureObject(tex_q);
}

void fuseTHuberQuadFit(iu::ImageGpu_32f_C1* u, iu::ImageGpu_32f_C1* u_, iu::ImageGpu_32f_C1* u0,
                     iu::ImageGpu_32f_C4* I1, iu::ImageGpu_32f_C4* I2,  iu::ImageGpu_32f_C4* I2_warped,
                     iu::ImageGpu_32f_C1* wx_gpu, iu::ImageGpu_32f_C1* wy_gpu, iu::ImageGpu_32f_C2* qf, iu::ImageGpu_32f_C2 *p,  iu::ImageGpu_32f_C2 *q,
                     iu::ImageGpu_32f_C4 *I1_padded, iu::ImageGpu_32f_C4 *I2_padded, int padval, ColorStereoNet *stereoNet,
                     float lambda_census, float C,
                     float disp_step, int iterations, int num_warps)
{
    int width = u->width();
    int height = u->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);
//    tvl1bounds_init_u_kernel<<<dimGrid,dimBlock,0,stream>>>(*u,*x1_gpu,*x2_gpu);
    // init p,q
    set_value_kernel<<<dimGrid,dimBlock,0,0>>>(make_float2(0.f,0.f),p->data(),p->width(),p->height(),p->stride());
    set_value_kernel<<<dimGrid,dimBlock,0,0>>>(make_float2(0.f,0.f),q->data(),q->width(),q->height(),q->stride());

    cudaTextureObject_t tex_I1, tex_I2;
    bindTexture(tex_I1,I1,cudaAddressModeClamp);
    bindTexture(tex_I2,I2,cudaAddressModeClamp);

    float L = sqrt(8);
    float tau = 1/L;
    float sigma = 1/L;
    // perform padding
    padColorImage(I1_padded,I1,padval);

    for (int warps=0; warps<num_warps; warps++)
    {
        warp_image_kernel <<< dimGrid, dimBlock,0,0 >>>(*I2_warped,*u,tex_I2);

        // call CNN
        padColorImage(I2_padded,I2_warped,padval);
        stereoNet->setDisparities(-disp_step*0.25,disp_step*0.25,disp_step*0.25,costvolume_refinement_);
        iu::TensorGpu_32f *d_out = stereoNet->predict(I1_padded, I2_padded);
        // perform fitting on CNN output
        do_fitting_kernel<<< dimGrid, dimBlock,0,0 >>>(*qf,*d_out,disp_step,lambda_census,false);
        iu::copy(u,u0);

        for (int iters=0; iters<iterations; iters++)
        {
            iu::copy(u,u_);
            cudaDeviceSynchronize();
            ttvqf_primal_kernel<<< dimGrid, dimBlock,0,0 >>>(*u, *u_, *u0, *qf, tex_p, tex_q, tau, disp_step);
            thuberqf_dual_kernel<<< dimGrid, dimBlock,0,0 >>>(*p, *q, *wx_gpu, *wy_gpu, tex_u_, sigma, C);
            cudaDeviceSynchronize();
        }
    }
    cudaDestroyTextureObject(tex_I1);
    cudaDestroyTextureObject(tex_I2);
}

void fuseQuadFitDirect(iu::ImageGpu_32f_C1* u, const ndarray_ref<float,3> &costvolume,
                     iu::ImageGpu_32f_C2* qf, float disp_min, float disp_step)
{
    int width = u->width();
    int height = u->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    // perform fitting on CRF output
    do_fitting_kernel_volume<<< dimGrid, dimBlock,0,0 >>>(*qf,*u,costvolume, disp_min, disp_step, 1.f);
    CudaCheckError();

    // perform argmin
    do_argmin_kernel_volume<<<dimGrid, dimBlock, 0, 0>>>(*u,*qf);
    CudaCheckError();
}


void leftRightCheck(iu::ImageGpu_32f_C1* device_out, iu::ImageGpu_32f_C1* device_left, iu::ImageGpu_32f_C1* device_right, float th)
{
    uint width = device_out->width();
    uint height = device_out->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    cudaTextureObject_t tex_left, tex_right;
    bindTexture(tex_left,device_left,cudaAddressModeClamp);
    bindTexture(tex_right,device_right, cudaAddressModeClamp);

    leftrightcheck_kernel<<<dimGrid,dimBlock>>>(*device_out, tex_left, tex_right, th);
//    if(!do_inf) {
//        interpolate_mismatch_kernel<<<dimGrid,dimBlock>>>(*device_out,disp_max);
//        interpolate_occlusion_kernel<<<dimGrid,dimBlock>>>(*device_out,disp_max);
        //median2d_kernel<<<dimGrid,dimBlock>>>(*device_out,2);
        //bilateral_filter_kernel<<<dimGrid,dimBlock>>>(*device_out,2,2,2);
//    }
    CudaCheckError();
    cudaDestroyTextureObject(tex_left);
    cudaDestroyTextureObject(tex_right);
}

void remapSolution(iu::ImageGpu_32f_C1*out, iu::ImageGpu_32s_C1* in, float disp_min, float disp_step, float disp_max)
{
    uint width = out->width();
    uint height = out->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    remapsolution_kernel<<<dimGrid,dimBlock>>>(*out,*in,disp_min,disp_step);
    cudaDeviceSynchronize();
}

void rgb2gray(iu::ImageGpu_32f_C4* rgb,iu::ImageGpu_32f_C1* gray)
{
//	iu::ImageGpu_32f_C4 rgba(rgb->size());
//	iu::convert(rgb, &rgba);

	dim3 dimBlock(32, 32);
    dim3 dimGrid(iu::divUp(rgb->width(), dimBlock.x),
                 iu::divUp(rgb->height(), dimBlock.y));

//    rgb_to_gray_kernel<<<dimGrid,dimBlock>>>(*rgb,*gray);
    rgb_to_gray_kernel<<<dimGrid,dimBlock>>>(*rgb,*gray);
}
} // namespace cuda
#endif
