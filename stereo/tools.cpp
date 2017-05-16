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

#include "tools.h"
#include <opencv2/core/core.hpp>
#include <png++/png.hpp>
#include <fstream>
#ifdef WITH_ITK
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkVTKImageIO.h>
#endif
#include <cnpy.h>

void saveMat(std::string filename, const iu::ImageGpu_32f_C1 *mat, bool png)
{
    iu::ImageCpu_32f_C1 in_cpu(mat->width(),mat->height());
    iu::copy(mat,&in_cpu);
    iu::Size<2> sz = mat->size();
    cv::Mat mat_32f(sz.height, sz.width, CV_32FC1, in_cpu.data(), in_cpu.pitch());
    if(png)
    {
        png::image< png::gray_pixel_16 > image(mat->width(),mat->height());
        for (int32_t v=0; v<mat->height(); v++) {
            for (int32_t u=0; u<mat->width(); u++) {
                image.set_pixel(u,v,(uint16_t)(std::max(mat_32f.at<float>(v,u)*256.0,0.0)));
            }
        }
        image.write(filename.c_str());
    }
    else
    {
        std::ofstream output;
        output.open(filename.c_str(),std::ios::out|std::ios::binary);
        output << "Pf" << std::endl;
        output << mat_32f.cols << " " << mat_32f.rows << std::endl << "-1.0" << std::endl;
        for(int row=mat_32f.rows-1;row>=0;row--)
        {
            output.write((char*)(mat_32f.data+mat_32f.step*row),mat_32f.cols*mat_32f.elemSize());
//            for(int col=0;col<mat_32f.cols;col++)
//            {
//                float val = mat_32f.at<float>(row,col);
//                output.write((char*)&val,sizeof(float));
//            }
        }
        output.close();
    }
}

void saveMat(std::string filename, const iu::ImageGpu_32s_C1 *mat)
{
    iu::ImageCpu_32s_C1 in_cpu(mat->width(),mat->height());
    iu::copy(mat,&in_cpu);
    iu::Size<2> sz = mat->size();
    cv::Mat mat_32s(sz.height, sz.width, CV_32S, in_cpu.data(), in_cpu.pitch());
    std::ofstream output;
    output.open(filename.c_str(),std::ios::out|std::ios::binary);
    output << "Pf" << std::endl;
    output << mat_32s.cols << " " << mat_32s.rows << std::endl << "-1.0" << std::endl;
    for(int row=0;row<mat_32s.rows;row++)
    {
        //output.write((char*)(mat_32f.data+mat_32f.step*row),mat_32f.cols*mat_32f.elemSize());
        for(int col=0;col<mat_32s.cols;col++)
        {
            float val = (float)mat_32s.at<int>(row,col);
            output.write((char*)&val,mat_32s.elemSize());
        }
    }
    output.close();
}

void saveMatPython2(std::string filename, iu::LinearDeviceMemory_32f_C1 *vol, int width, int height, int depth)
{
    iu::LinearHostMemory_32f_C1 in_cpu(vol->size());
    iu::copy(vol,&in_cpu);
    const unsigned int shape[] = {height,width,depth};
    cnpy::npy_save(filename,in_cpu.data(),shape,3);
    //delete in_cpu;
}

void saveMatPython(std::string filename, iu::ImageGpu_32f_C1 *mat)
{
    //iu::ImageCpu_32f_C1 in_cpu(mat->size());
    float *cpu_data = new float[mat->size().width*mat->size().height];
    iu::ImageCpu_32f_C1 in_cpu(cpu_data,mat->size().width,mat->size().height,mat->size().width*sizeof(float),true);
    iu::copy(mat,&in_cpu);
    const unsigned int shape[] = {mat->height(),mat->width()};
    cnpy::npy_save(filename,in_cpu.data(),shape,2);
    delete cpu_data;
}

void saveMat(std::string filename, const iu::LinearDeviceMemory_32f_C1 *vol, int width, int height, int depth)
{
#ifdef WITH_ITK
    typedef float PixelType;
    typedef itk::Image<PixelType, 3> ImageType;
    typedef itk::ImageFileWriter<ImageType> WriterType;
    typedef itk::VTKImageIO ImageIOType;

    ImageType::Pointer v_volume = ImageType::New();
    ImageIOType::Pointer vtkIO = ImageIOType::New();

    // first we need to get the volume to the cpu
//    short2 *cpu_volume = new short2[fusion.integration.size.x*fusion.integration.size.y*fusion.integration.size.z];
//    cudaMemcpy((void*)cpu_volume,(void*)fusion.integration.data,
//               fusion.integration.size.x*fusion.integration.size.y*fusion.integration.size.z*sizeof(short2),cudaMemcpyDeviceToHost);
    iu::LinearHostMemory_32f_C1 cpu_volume(vol->size());
    iu::copy(vol,&cpu_volume);

    ImageType::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = depth;
    v_volume->SetRegions(size);

    ImageType::SpacingType spacing;

    spacing[0] = 1;
    spacing[1] = 1;
    spacing[2] = 1;
    v_volume->SetSpacing(spacing);

    v_volume->Allocate();

    float* iter = cpu_volume.data();
    for (unsigned int y=0; y<size[1]; y++) {
        for (unsigned int x=0; x<size[0]; x++) {
            for (unsigned int z = 0; z < size[2]; z++) {
                ImageType::IndexType idx;
                idx[0] = x;
                idx[1] = y;
                idx[2] = z;
                v_volume->SetPixel(idx, *iter);
                ++iter;
            }
        }
    }
//    delete cpu_volume;


    WriterType::Pointer writer1 = WriterType::New();
    writer1->SetImageIO(vtkIO);
    writer1->SetInput( v_volume );
    writer1->SetFileName( filename.c_str() );
    try
    {
        writer1->Update();
    }
    catch( itk::ExceptionObject & excp )
    {
        std::cerr << "Error writing vtk " << std::endl;
        std::cerr << excp << std::endl;
        return;
    }
#endif
}
