CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(cudnn_stereo)
###############################################################################
# Set Compiler Options
###############################################################################
#SET(CMAKE_BUILD_TYPE Release) # (DEBUG/MINSIZEREL/RELWITHDEBINFO/RELEASE)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -O3")
OPTION(BUILD_RELEASE "Build with Release Code" ON)

if(BUILD_RELEASE)
    # for openmp
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    add_definitions("-O3 -DPARALLEL -ffast-math --std=c++11")
    set(CMAKE_BUILD_TYPE Release)
else(BUILD_RELEASE)
    add_definitions("-ggdb")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-G -keep")
    set(CMAKE_BUILD_TYPE Debug)
endif(BUILD_RELEASE)


#SET(CMAKE_BUILD_TYPE Debug) # (DEBUG/MINSIZEREL/RELWITHDEBINFO/RELEASE)
#SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#SET(BUILD_RELEASE OFF)

###############################################################################
# Find necessary libraries
###############################################################################
SET(EXTERN_LIBS "")

# MobileVision ImageUtilities
# -----------------------------------------------------------------------------
SET(ImageUtilities_DIR $ENV{IMAGEUTILITIES_ROOT})
set(IMAGEUTILITIES_PREFER_SHARED_LIBRARIES true)

find_package(ImageUtilities REQUIRED COMPONENTS iucore iuio iumath)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir1='${dir}'")
endforeach()

cuda_include_directories(${IMAGEUTILITIES_INCLUDE_DIR})
include_directories(${IMAGEUTILITIES_INCLUDE_DIR})
SET(EXTERN_LIBS ${EXTERN_LIBS} ${IMAGEUTILITIES_LIBRARIES})

include(../cmake/compiler_settings.cmake.txt)

# SlackProp
#------------------------------------------------------------------------------
#SET(SlackProp_DIR $ENV{SLACKPROP_ROOT})
SET(SlackProp_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/slackprop)
INCLUDE_DIRECTORIES(${SlackProp_DIR}/src)

SET(EXTERN_LIBS ${EXTERN_LIBS} ${SlackProp_DIR}/lib/libslack_prop.so)

# cnpy
# -----------------------------------------------------------------------------
SET(CNPY_INCLUDE_DIR "" CACHE FILEPATH "Path to cnpy include")

IF(NOT EXISTS /usr/local/include/cnpy.h)
	IF(CNPY_INCLUDE_DIR STREQUAL "")
	  MESSAGE(SEND_ERROR 
	  "Can't find cnpy.h on system path. Install cnpy to system or set additional include directory!")
	ENDIF()
ENDIF()

SET(CNPY_LIB_DIR "" CACHE FILEPATH "Path to cnpy lib")
IF(NOT EXISTS /usr/local/lib/libcnpy.so)
	IF(CNPY_LIB_DIR STREQUAL "")
	  MESSAGE(SEND_ERROR 
	  "Can't find libcnpy.so on system path. Install cnpy to system or set additional library directory!")
	ENDIF()
ENDIF()


INCLUDE_DIRECTORIES(${CNPY_INCLUDE_DIR})
LINK_DIRECTORIES(${CNPY_LIB_DIR})

SET(EXTERN_LIBS ${EXTERN_LIBS} cnpy)



##-----------------------------------------------------------------------------
## Boost
find_package(Boost COMPONENTS program_options filesystem system REQUIRED)

# CUDA related stuff
# -----------------------------------------------------------------------------
#FIND_PACKAGE(CUDA REQUIRED)
#INCLUDE(FindCUDA)
LINK_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)
#SET(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G -keep -lineinfo")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

# add cudnn to cuda libraries
SET(CUDA_LIBRARIES ${CUDA_LIBRARIES} cudnn)

#MESSAGE("nvcc flags ${CUDA_NVCC_FLAGS}")
#MESSAGE("nvcc debug flags ${CUDA_NVCC_FLAGS_DEBUG}")

###############################################################################
# Specify source files and headers for compilation
###############################################################################
#FILE(GLOB SOURCES "src/*.cpp) allows wildcard addition


SET(SOURCES
	stereonet.cpp
	colorstereonet.cpp
	utils.cpp
	utils.cu
	operators/operator.cpp
	operators/bias.cpp
	operators/activation.cpp
	operators/softmax.cpp
	operators/convolution.cpp
	operators/stereocorrelation.cpp
	operators/stereocorrelation.cu
	operators/slackprop.cpp
)

SET(HEADERS
	stereonet.h
	colorstereonet.h
	error_util.h
	utils.h
	utils.cuh
	operators/operator.h
	operators/bias.h
	operators/activation.h
	operators/softmax.h
	operators/convolution.h
	operators/stereocorrelation.h
	operators/stereocorrelation.cuh
	operators/slackprop.h
)

SET (SRC_STEREO_LIB
  stereo.h
  stereo.cpp
  stereo.cu
  stereo.cuh

  tools.h
  tools.cpp
)

SET ( SRC_STEREO_IMG
  stereo_img.cpp
)

###############################################################################
# Specify what and how to build
###############################################################################
CUDA_ADD_LIBRARY(cudnn_stereo SHARED ${HEADERS} ${SOURCES}) 
TARGET_LINK_LIBRARIES(cudnn_stereo ${CUDA_LIBRARIES} ${EXTERN_LIBS} cnpy) 

cuda_add_library(stereo SHARED ${SRC_STEREO_LIB})
TARGET_LINK_LIBRARIES(stereo ${IMAGEUTILITIES_LIBRARIES} ${LIBS}
${OpenCV_LIBRARIES} ${OpenCV_LIBS} cudnn_stereo ${SLACK_PROP_LIBRARY} ${ITK_LIBRARIES} cnpy png)

cuda_add_executable( stereo_img ${SRC_STEREO_IMG})
TARGET_LINK_LIBRARIES(stereo_img  stereo ${IMAGEUTILITIES_LIBRARIES} ${Boost_LIBRARIES})

MESSAGE("link dirs ${EXTERN_LIBS}")


