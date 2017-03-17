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

#ifndef STEREO_CUH
#define STEREO_CUH

#include <cuda_runtime.h>
//#include <cutil_inline.h>
//#include <cutil_math.h>
#include "iu/iucore.h"
#include "iu/iucutil.h"
#include "iu/ndarray/ndarray.h"
#include "iu/ndarray/ndarray_iu.h"
//#include "iumath.h"
class GrayStereoNet;
class ColorStereoNet;

namespace cuda {
    void calccostvolumecensus(iu::ImageGpu_32f_C1* I1, iu::ImageGpu_32f_C1* I2,
                        float lambda, int filter_size, float disp_min, float disp_step, float disp_max,
                        iu::LinearDeviceMemory_32f_C1* costvolume, cudaStream_t stream);

    iu::TensorGpu_32f *calccostvolumeCNN(iu::ImageGpu_32f_C1 *I1, iu::ImageGpu_32f_C1 *I1_padded, iu::ImageGpu_32f_C1 *I2, iu::ImageGpu_32f_C1 *I2_padded,
    		GrayStereoNet *stereoNet, int padval);
    iu::TensorGpu_32f *calccostvolumeColorCNN(iu::ImageGpu_32f_C4 *I1, iu::ImageGpu_32f_C4 *I1_padded, iu::ImageGpu_32f_C4 *I2, iu::ImageGpu_32f_C4 *I2_padded,
        		ColorStereoNet *stereoNet, int padval);
    void destroyCNN(void);
    void computeArgmin(iu::ImageGpu_32f_C1* device_out, iu::LinearDeviceMemory_32f_C1* costvolume, float disp_min, float disp_step);
    void leftRightCheck(iu::ImageGpu_32f_C1* device_out, iu::ImageGpu_32f_C1* device_left, iu::ImageGpu_32f_C1* device_right, float th);
    void prepareOcclusionInpainting(iu::ImageGpu_32f_C1* inpainted_out, iu::ImageGpu_8u_C1* occlusionmask_out, iu::ImageGpu_32f_C1* disp_lr_in,
                                    iu::ImageGpu_32f_C1* disp_rl_in, float th, float disp_min, float disp_step, float disp_max, int filter_size);
    void filter(iu::ImageGpu_32f_C1* device_in,iu::ImageGpu_32f_C1* device_out, bool x, bool forward, float alpha, float beta);
    void calcEdgeImage(iu::ImageGpu_32f_C1* I1, iu::ImageGpu_32f_C1 *temp, iu::ImageGpu_32f_C1* wx, iu::ImageGpu_32f_C1* wy, float alpha, float beta, float lambda);

    void prepareFuseTTVL1(iu::ImageGpu_32f_C1* u,iu::ImageGpu_32f_C1* u_ ,iu::ImageGpu_32f_C2* p,iu::ImageGpu_32f_C2* q);
    void unprepareFuseTTVL1();

    void fuseTHuberQuadFit(iu::ImageGpu_32f_C1* u, iu::ImageGpu_32f_C1 *u_, iu::ImageGpu_32f_C1 *u0, iu::ImageGpu_32f_C4 *I1, iu::ImageGpu_32f_C4 *I2, iu::ImageGpu_32f_C4 *I2_warped,
                          iu::ImageGpu_32f_C1* wx_gpu, iu::ImageGpu_32f_C1* wy_gpu, iu::ImageGpu_32f_C2 *qf,  iu::ImageGpu_32f_C2 *p, iu::ImageGpu_32f_C2 *q,
                          iu::ImageGpu_32f_C4 *I1_padded, iu::ImageGpu_32f_C4 *I2_padded, int padval, ColorStereoNet *stereoNet, float lambda_census, float C, float disp_step, int iterations, int warps);
    void fuseTHuberQuadFit(iu::ImageGpu_32f_C1* u, iu::ImageGpu_32f_C1 *u_, iu::ImageGpu_32f_C1 *u0, iu::ImageGpu_32f_C1 *I1, iu::ImageGpu_32f_C1 *I2, iu::ImageGpu_8u_C1 *occlusion_mask,
                          iu::ImageGpu_32f_C1* wx_gpu, iu::ImageGpu_32f_C1* wy_gpu, iu::ImageGpu_32f_C2 *qf,  iu::ImageGpu_32f_C2 *p, iu::ImageGpu_32f_C2 *q,
                          iu::ImageGpu_32f_C1 *I1_padded, iu::ImageGpu_32f_C1 *I2_padded, int padval, GrayStereoNet *stereoNet,
                          int filter_size, float lambda_census, float C, float disp_step, int iterations, int warps);

    void fuseQuadFitDirect(iu::ImageGpu_32f_C1* u, const ndarray_ref<float,3> &costvolume,
                           iu::ImageGpu_32f_C2* qf, float disp_min, float disp_step);

    void remapSolution(iu::ImageGpu_32f_C1*out, iu::ImageGpu_32s_C1* in, float disp_min, float disp_step, float disp_max);
    void rgb2gray(iu::ImageGpu_32f_C4* rgb,iu::ImageGpu_32f_C1* gray);
} // namespace cuda
#endif
