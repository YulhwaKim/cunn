#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#define BLOCK_SIZE 16

// Shape of data :  OUT (nOUtputPlane, nOutSpatial); IN (nIn, nOutSpatial); W (nOutputPlane, nIn)
template <typename T, typename AccumT>
__global__ void cunn_CrossbarSpatialConvolutionWvar_updateOutput_frame_kernel(
  T *OUT, T *IN, T *W, T *VarP, T *VarM, int accumN, long nIn, long nOutSpatial, long nOutputPlane)
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
  __shared__ T VarPs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T VarMs[BLOCK_SIZE][BLOCK_SIZE];
  
  // each thread do the vector-vector multiplication
  // thus, have to repeat on size_vector(nIn) elements
  AccumT psum = 0;
  AccumT output_temp = 0;
  unsigned int accumCount = 0;
  long i = 0;
  while(i < nIn){
    // copy the data from global memory to shared memory
    INs[ty][tx] = IN[(i+tx)*nOutSpatial + INcol];
    Ws[ty][tx] = W[Wrow*nIn + (i+ty)];
    VarPs[ty][tx] = VarP[Wrow*nIn + (i+ty)];
    VarMs[ty][tx] = VarM[Wrow*nIn + (i+ty)];
    __syncthreads();
    
    // compute element-size multiplication
    for(unsigned int j=0; j<BLOCK_SIZE; j++) {
      // multiplication
//       if((Wrow < nOutputPlane) && (INcol < nOutSpatial))
//       printf("INs : %.1f\n", INs[ty][j]);
      T temp = INs[ty][j] * Ws[j][tx];
      // Variation modeling
      temp = (temp > 0)? temp + VarPs[j][tx] : temp + VarMs[j][tx];
      // Accumulation
      psum += temp;
      accumCount += 1;
      // digitize psum
      if(accumCount >= accumN) {
        // quantize psum
        if (accumN == 1) 
          psum = (psum >= 0)? 1 : -1;
        else {
          psum = roundf(psum/2)*2;
          // clamping
          psum = (psum > accumN)? accumN : psum;
          psum = (psum < (-1)*accumN)? (-1)*accumN : psum;
        }
        // update output_temp
        output_temp += ScalarConvert<AccumT, T>::to(psum);
        // update or reset states
        psum = 0;
        accumCount = 0;
      }
    }
    __syncthreads();
    i += BLOCK_SIZE;
  }
  // update outputs
  if((Wrow < nOutputPlane) && (INcol < nOutSpatial)) // shut down kernels that are not in the range
    OUT[Wrow*nOutSpatial + INcol] = ScalarConvert<AccumT, T>::to(output_temp);
}


#include "generic/CrossbarSpatialConvolutionWvar.cu"
#include "THCGenerateFloatTypes.h"
