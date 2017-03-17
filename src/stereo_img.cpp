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

#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>

#include <iu/iucore.h>
#include <iu/iuio.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "stereo.h"
#include "tools.h"

//#include "opencv2/imgproc/imgproc.hpp"

using std::string;
using std::cout;
using std::endl;

namespace po = boost::program_options;

constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

int main(int argc, char *argv[])
{
    // parameter parsing using boost program_options
    po::options_description general("General options");
    general.add_options()
        ("help", "produce help message")
        ("im0", po::value<std::string>(), "Left Image")
        ("im1", po::value<std::string>(), "Right Image")
        ("disp-min", po::value<double>()->default_value(1),"Minimum disparity")
        ("disp-step", po::value<double>()->default_value(1),"Disparity step")
        ("disp-max", po::value<double>()->default_value(128),"Maximum disparity")
        ("matching", po::value<std::string>()->default_value("CENSUS"), "Matching method. Possible options: CENSUS, COLORCNN, GRAYCNN, CONCATCNN, COLORPAIR, LOAD")
        ("volume", po::value<std::string>()->default_value(""), "Volume file when matching method is set to LOAD")
        ("inference", po::value<std::string>()->default_value("CRF"), "Inference method. Possible options: CRF, ARGMAX, NONE")
        ("refinement", po::value<std::string>()->default_value("NONE"), "Continuous refinement. Possible options: NONE, THuberQuad, QuadDirect")
        ("alpha", po::value<double>()->default_value(20.0),"Parameter for edge image calculation")
        ("beta", po::value<double>()->default_value(0.8),"Parameter for edge image calculation")
        ("gpu,g", po::value<int>()->default_value(0),"GPU to use")
        ("output-file,o", po::value<std::string>()->default_value("./output"),"Base name (without extension), where all the files will be written")
        ("config-file", po::value<std::string>(),"Provide all options in a convenient config file")
    ;
    po::options_description cnn("CNN + CNNPAIR options");
    cnn.add_options()
        ("num-layers", po::value<int>()->default_value(7),"Number of layers for CNN matching")
        ("parameter-file", po::value<std::string>()->default_value("params"), "Parameters for CNN matching without extension. It will be calculated as <parameter_file>_<num_layers>.npz")
        ("with-xy", po::value<bool>()->default_value(false), "Use x- and y-coordinates (only possible for KITTI")
    ;
    po::options_description census("CENSUS options");
    census.add_options()
        ("filter-size", po::value<int>()->default_value(7),"Filter size for Census matching")
    ;
    po::options_description crf("CRF options");
    crf.add_options()
        ("lambda", po::value<double>()->default_value(0.4),"Weighting of regularization term")
        ("L1", po::value<double>()->default_value(1),"Parameter for CRF inference")
        ("L2", po::value<double>()->default_value(12),"Parameter for CRF inference")
        ("delta", po::value<double>()->default_value(2),"Parameter for CRF inference")
        ("crf-iterations", po::value<int>()->default_value(4),"Parameter for CRF inference")
    ;
    po::options_description ref("Refinement options");
    ref.add_options()
        ("ref-iterations", po::value<int>()->default_value(10),"Refinement iterations")
        ("warps", po::value<int>()->default_value(2),"Refinement warps")
    ;
    po::options_description desc("Allowed options");
    desc.add(general).add(cnn).add(census).add(crf).add(ref);
    po::positional_options_description p;
    p.add("im0", 1);
    p.add("im1", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
    // optionally read config file
    if(vm.count("config-file")) {
        std::ifstream ifs(vm["config-file"].as<string>().c_str(), std::ifstream::in);
        if(ifs.good())
            po::store(po::parse_config_file(ifs,desc),vm);
    }
    po::notify(vm);

    if (!vm.count("im0") || !vm.count("im1")) {
        cout << desc  << "\n";
        return 1;
    }

    // Set GPU first
    cudaSetDevice(vm["gpu"].as<int>());

    string fname_I1 = vm["im0"].as<string>();
    string fname_I2 = vm["im1"].as<string>();
    string fname_out= vm["output-file"].as<string>();
    string fname_params = vm["parameter-file"].as<string>();
    
    float min_disp = vm["disp-min"].as<double>();
    float max_disp = vm["disp-max"].as<double>();
    float disp_step = vm["disp-step"].as<double>();
    float lambda_cost = vm["lambda"].as<double>();
    int refinement_iterations = vm["ref-iterations"].as<int>();
    int crf_iterations = vm["crf-iterations"].as<int>();
    float L1 = vm["L1"].as<double>();
    float L2 = vm["L2"].as<double>();
    float delta = vm["delta"].as<double>();
    float alpha = vm["alpha"].as<double>();
    float beta = vm["beta"].as<double>();
    int filter_size = vm["filter-size"].as<int>();
    int num_layers = vm["num-layers"].as<int>();
    int warps = vm["warps"].as<int>();

    bool with_xy = vm["with-xy"].as<bool>();

	Stereo stereo;
  stereo.setAllowCnnGc(true);
	stereo.setWithXy(with_xy);

	iu::ImageCpu_32f_C4 *I1 = iu::imread_32f_C4(fname_I1);
	iu::ImageCpu_32f_C4 *I2 = iu::imread_32f_C4(fname_I2);
	stereo.initialize(I1->size(), min_disp, disp_step, max_disp);
    stereo.setVerbose(false);

    iu::imsave(I1, "/tmp/out.png");
    iu::imsave(I2, "/tmp/out1.png");

    switch(str2int(vm["matching"].as<std::string>().c_str()))
    {
        case str2int("COLORCNN"):
            stereo.setNumLayers(num_layers);
            stereo.setParameterFile(fname_params);
            stereo.setMatchingFunction(COLORCNN);
            break;
        case str2int("CONCATCNN"):
			stereo.setNumLayers(num_layers);
			stereo.setParameterFile(fname_params);
			stereo.setMatchingFunction(CONCATCNN);
			break;
        case str2int("GRAYCNN"):
			stereo.setNumLayers(num_layers);
			stereo.setParameterFile(fname_params);
			stereo.setMatchingFunction(GRAYCNN);
			break;
        case str2int("GRAYPAIR"):
            alpha = 0.f; // to ignore the edge term
            stereo.setNumLayers(num_layers);
            stereo.setParameterFile(fname_params);
            stereo.setMatchingFunction(GRAYCNN_PAIRWISE);
            break;
        case str2int("COLORPAIR"):
			alpha = 0.f; // to ignore the edge term
			stereo.setNumLayers(num_layers);
			stereo.setParameterFile(fname_params);
			stereo.setMatchingFunction(COLORCNN_PAIRWISE);
			break;
        case str2int("CONCATPAIR"):
			alpha = 0.f; // to ignore the edge term
			stereo.setNumLayers(num_layers);
			stereo.setParameterFile(fname_params);
			stereo.setMatchingFunction(GRAYCNN_PAIRWISE);
			break;
        case str2int("CENSUS"):
            stereo.setMatchingFunction(CENSUS);
            stereo.setFilterSize(filter_size);
            break;
        case str2int("LOAD"):
            stereo.setMatchingFunction(LOAD);
            stereo.loadVolumeFromFile(vm["volume"].as<string>());
            break;
        default:
            cout << "Unknown matching function (" << str2int(vm["matching"].as<std::string>().c_str()) << ")" << endl;
            return 1;
            break;
    }

    stereo.setColorImages(*I1, *I2);

	switch(str2int(vm["inference"].as<std::string>().c_str()))
    {
        case str2int("CRF"):
            stereo.computeSlackPropagation(L1,L2,delta, crf_iterations,alpha, beta,lambda_cost,true);
            switch (str2int(vm["refinement"].as<std::string>().c_str())) {
                case str2int("THuberQuad"):
                    stereo.fuseTHuberQuadFit(lambda_cost,L2,refinement_iterations, warps);
                    break;
                case str2int("QuadDirect"):
                    stereo.fuseQuadFitDirect();
                    break;
                case str2int("NONE"):
                    break;
                default:
                    cout << "Unkonwn refinement method" << endl;
                    return 1;
                    break;
            }
            saveMat(fname_out+".png",stereo.getOutput(),true);
            saveMat(fname_out+".pfm",stereo.getOutput(),false);
            saveMat(fname_out+"_wx.pfm",stereo.getWx(alpha,beta,true),false);
            saveMat(fname_out+"_wy.pfm",stereo.getWy(alpha,beta,true),false);
            break;
        case str2int("ARGMAX"):
            stereo.computeArgMax(true);
            saveMat(fname_out+".png",stereo.getOutput(),true);
            saveMat(fname_out+".pfm",stereo.getOutput(),false);
            break;
        case str2int("NONE"):
            stereo.computeVolume(true);
            saveMatPython(fname_out+"_wx.npy",stereo.getWx(alpha,beta,true));
            saveMatPython(fname_out+"_wy.npy",stereo.getWy(alpha,beta,true));
            saveMatPython2(fname_out + ".npy", stereo.getVolume(), I1->width(),I1->height(),stereo.getDepth());
            saveMat(fname_out+".vtk",stereo.getVolume(),I1->width(),I1->height(),stereo.getDepth());
            break;
        default:
            cout << "Unkonwn inference method" << endl;
            return 1;
            break;
    }


    delete I1;
    delete I2;

    
    return EXIT_SUCCESS;
    
}
