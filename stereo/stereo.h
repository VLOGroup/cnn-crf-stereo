// This file is part of cnn-crf-stereo.
//
// Copyright (C) 2017 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// cnn-crf-stereo is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// cnn-crf-stereo is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef STEREO_H
#define STEREO_H

#include "opencv2/core/core.hpp"
#include "iu/iucore.h"
#include "slack_prop.h"
#include "iu/ndarray/ndarray_iu.h"

#include "stereonet.h"


enum MatchingFunction {
    COLORCNN,
    GRAYCNN,
    CONCATCNN,
    GRAYCNN_PAIRWISE,
    COLORCNN_PAIRWISE,
    CONCATCNN_PAIRWISE,
    CENSUS,
    LOAD
};

class Stereo {
public:

    Stereo();
    virtual ~Stereo();
    void initialize(const iu::Size<2> size, float disp_min_, float disp_step_, float disp_max_);
    void setImages(const iu::ImageCpu_32f_C1 &img1, const iu::ImageCpu_32f_C1 &img2);
    void setColorImages(const iu::ImageCpu_32f_C4 &img1, const iu::ImageCpu_32f_C4 &img2);
    void computeSlackPropagation(float C1, float C2, float delta, int iterations, float alpha, float beta, float lambda_reg, bool lr);
    void computeArgMax(bool lr);
    void computeVolume(bool lr);
    void computeWeights(float alpha, float beta, float lambda, bool lr);
    void setMatchingFunction(MatchingFunction function);
    void setNumLayers(int num_layers);
    void setParameterFile(std::string base_name);
    void setFilterSize(int filter_size){filter_size_ = filter_size;}
    bool loadVolumeFromFile(std::string filename);
    void setVerbose(bool val) {verbose_=val;}

    void fuseTHuberQuadFit(float lambda_reg, float C, int iterations=100, int warps = 4);
    void fuseQuadFitDirect(void);

    iu::ImageGpu_32f_C1 *getOutput();
    iu::LinearDeviceMemory_32f_C1 *getVolume();
    iu::ImageGpu_32f_C1* getWx(float alpha, float beta,bool left=true);
    iu::ImageGpu_32f_C1* getWy(float alpha, float beta,bool left=true);

    void performLeftRightCheck(iu::ImageGpu_32f_C1* out, iu::ImageGpu_32f_C1* left, iu::ImageGpu_32f_C1* right, float th);
    iu::ImageGpu_32f_C1 *getLeftImage(){return img1;}
    iu::ImageGpu_32f_C4 *getLeftColorImage(){return img1_color;}
    iu::ImageGpu_32f_C1 *getRightImage(){return img2;}
    iu::ImageGpu_32f_C4 *getRightColorImage(){return img2_color;}
    iu::ImageGpu_32f_C1 *getLeftEdgeImage(){return wx;}
    iu::ImageGpu_32f_C1 *getRightEdgeImage(){return wy;}

    int getDepth() const;
    void setAllowCnnGc(bool value)
    {
    	allowCnnGc = value;
    	if(stereoNet != NULL)
    		stereoNet->setAllowGc(value);
    }
    void setWithXy(bool value) { with_xy = value; }

private:
    bool verbose_;
    void destroyCNN(void);
    void setDisparityRange(float disp_min_,float disp_step_, float disp_max_);
    int log2i(int num);
    //iu::ImageGpu_32f_C1 *getSlackPropOutput();
     iu::ImageGpu_32s_C1 *getIntegerOutput();
    int width_,height_,depth_;
    float disp_min_, disp_step_, disp_max_;
    iu::ImageGpu_32f_C1  *img1;
    iu::ImageGpu_32f_C1  *img2;
    iu::ImageGpu_32f_C4  *img1_color;
    iu::ImageGpu_32f_C4  *img2_color;
    iu::LinearDeviceMemory_32f_C1 *costvolume_;
    //iu::LinearDeviceMemory_32f_C1 *costvolume_cnn_;
    int filter_size_;
    int num_cnn_layers_;
    std::string cnn_parameter_file_base_name_;
    MatchingFunction matching_function_;
    // Stuff for inpainting
    bool do_inpainting;
    iu::ImageGpu_8u_C1 *occlusion_mask;

    iu::ImageGpu_32f_C1  *wx;
    iu::ImageGpu_32f_C1  *wy;
    iu::ImageGpu_32f_C1  *out0;
    iu::ImageGpu_32s_C1  *out_slack;
    iu::ImageGpu_32f_C1  *out;
    iu::ImageGpu_32f_C1  *out_;
    iu::ImageGpu_32f_C2  *p;
    iu::ImageGpu_32f_C2  *q;
    iu::ImageGpu_32f_C2  *qf;

    slack_prop_2D_alg *solver;

    // CNN
    StereoNet* stereoNet = NULL;
    iu::TensorGpu_32f *d_stereoNetMatchingOutput = NULL;
    iu::TensorGpu_32f *d_pairwiseOut = NULL;
    iu::ImageGpu_32f_C1 *I1_padded = NULL;
    iu::ImageGpu_32f_C1 *I2_padded = NULL;
    iu::ImageGpu_32f_C4 *I1_padded_color = NULL;
    iu::ImageGpu_32f_C4 *I2_padded_color = NULL;
    iu::ImageGpu_32f_C4 *I2_warped_color = NULL;
    int padval = 0;
    int rect_corr = 0;
    bool allowCnnGc = false;
    bool with_xy = false;

};







#endif //TVLINPROX_H
