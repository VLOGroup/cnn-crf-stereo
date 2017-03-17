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

#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#include <iu/iucore.h>
#include <GL/glu.h>

// OpenGL error check macro
#define GL_CHECK_ERROR() do { \
    GLenum err = glGetError(); \
    if(err != GL_NO_ERROR)     \
        printf("OpenGL error: %s\n File: %s\n Function: %s\n Line: %d\n", gluErrorString(err), __FILE__, __FUNCTION__, __LINE__); \
    } while(0)

// CUDA error check macro
#define CUDA_CHECK_ERROR() do { \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess)     \
        printf("CUDA error: %s:%s\n File: %s\n Function: %s\n Line: %d\n", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __FUNCTION__, __LINE__); \
    } while(0)



using std::string;

void saveMat(std::string filename, const iu::ImageGpu_32f_C1 *mat, bool png=false);
void saveMat(std::string filename, const iu::ImageGpu_32s_C1 *mat);
void saveMat(std::string filename, const iu::LinearDeviceMemory_32f_C1 *vol, int width, int height, int depth);
void saveMatPython(std::string filename, iu::ImageGpu_32f_C1 *mat);
void saveMatPython2(std::string filename, iu::LinearDeviceMemory_32f_C1 *vol, int width, int height, int depth);

#endif // TOOLS_H

