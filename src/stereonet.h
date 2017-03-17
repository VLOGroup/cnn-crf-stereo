// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Patrick Knöbelreiter <knoebelreiter at icg dot tugraz dot at>
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

#pragma once
#include <vector>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>

#include "operators/operator.h"
#include "error_util.h"

class StereoNet
{
  public:
	StereoNet(unsigned int numLayers, unsigned int in, unsigned int ic, unsigned int ih, unsigned int iw);
    virtual ~StereoNet();

    void setDisparities(float min_disp, float max_disp, float step, iu::TensorGpu_32f *d_out=NULL);

    void setAllParams(std::string npzPath);
    float getParams(std::vector<iu::TensorGpu_32f> &params);

    void loadParams(std::string &filename);
    void saveParams(std::string &filename);

    void setVerbose(bool val) { m_verbose = val; }

    virtual void initNet(float min_disp=0, float max_disp=0, float step=0, int rect_corr=0,
    		iu::TensorGpu_32f *d_unaryOut=NULL, iu::TensorGpu_32f *d_pairwiseOut=NULL);

    void setAllowGc(bool value) { m_allowGc = value; }


  protected:
    iu::TensorGpu_32f *performPrediction(iu::TensorGpu_32f *d_left, iu::TensorGpu_32f *d_right);

    cudnnHandle_t m_cudnnHandle;

    std::vector<Operator*> m_leftOps;
    std::vector<Operator*> m_rightOps;
    std::vector<Operator*> m_lrOps;

    std::vector<Operator*> m_pairwiseOps;

    int m_numLayers;
    int m_in;
    int m_ic;
    int m_ih;
    int m_iw;

    bool m_verbose;
    bool m_initialized;
    bool m_allowGc;

    iu::TensorGpu_32f *m_pwInput;

};
