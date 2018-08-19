#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#define BLOCK_SIZE 16

// Shape of data :  OUT (nOUtputPlane, nOutSpatial, nPsum); IN (nIn, nOutSpatial); W (nOutputPlane, nIn)
template <typename T, typename AccumT>
__global__ void cunn_CrossbarSpatialConvolution_updateOutput_kernel(
  T *OUT, T *IN, T *W, int accumN, long nIn, long nOutSpatial, long nOutputPlane, long nPsum)
{
  // index of output matrix
  int Wrow = blockIdx.x * blockDim.x + threadIdx.x;
  int INcol = blockIdx.y * blockDim.y + threadIdx.y;
  
  // thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // used BLOCK_SIZE as the TILE size
  __shared__ T INs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T Ws[BLOCK_SIZE][BLOCK_SIZE];
  
  // each thread do the vector-vector multiplication
  // thus, have to repeat on size_vector(nIn) elements
  AccumT temp = 0;
  unsigned int accumCount = 0;
  long OUTcol = 0;
  long i = 0;
  while(i < nIn){
    // copy the data from global memory to shared memory
    INs[ty][tx] = IN[(i+tx)*nOutSpatial + INcol];
    Ws[ty][tx] = W[Wrow*nIn + (i+ty)];
    __syncthreads();
    
    // compute element-size multiplication
    for(unsigned int j=0; j<BLOCK_SIZE; j++) {
      // do the accumulation
      temp += INs[ty][j] * Ws[j][tx];
      accumCount += 1;
      if(accumCount >= accumN) {
        // update outputs
        if((Wrow < nOutputPlane) && (INcol < nOutSpatial) && (OUTcol < nPsum)) { // shut down kernels that are not in the range
          OUT[Wrow*nOutSpatial*nPsum + INcol*nPsum + OUTcol] = ScalarConvert<AccumT, T>::to(temp);
        }
        // update or reset states
        OUTcol += 1; 
        temp = = 0;
        accumCount = 0;
      }
    }
    __syncthreads();
    i += BLOCK_SIZE;
  }
}


#include "generic/CrossbarSpatialConvolution.cu"
#include "THCGenerateFloatTypes.h"
