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

