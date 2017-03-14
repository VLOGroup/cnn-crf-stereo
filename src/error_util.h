#pragma once
#include "cudnn.h"
#include "stdio.h"
#include "stdlib.h"

#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cnnCudaSafeCall( err, __FILE__, __LINE__ )
#define cudnnSafeCall( err ) __cnnCudnnSafeCall( err, __FILE__, __LINE__ )

inline void __cnnCudaSafeCall( cudaError_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cnnCudnnSafeCall( cudnnStatus_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if(err != CUDNN_STATUS_SUCCESS)
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudnnGetErrorString( err ) );
		        exit( -1 );
	}
#endif

	return;
}


