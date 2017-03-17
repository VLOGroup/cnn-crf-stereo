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


