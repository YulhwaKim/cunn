#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#define BLOCK_SIZE 16

template <typename T, typename AccumT>
__global__ void cunn_CrossbarSpatialConvolution_updateOutput_kernel(
  T *OUT, T *IN, T *W, int accumN, long nIn, long nOutSpatial)
{

}


#include "generic/CrossbarSpatialConvolution.cu"
#include "THCGenerateFloatTypes.h"
