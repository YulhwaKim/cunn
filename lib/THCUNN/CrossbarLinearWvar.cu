#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

#define BLOCK_SIZE 16

template <typename T, typename AccumT>
__global__ void cunn_CrossbarLinearWvar_updateOutput_kernel(
  T *OUT, T *IN, T *W, T *VarP, T *VarM, int accumN, long nBatch, long nIn, long nOut)
{
  // index of output matrix
  int Wrow = blockIdx.x * blockDim.x + threadIdx.x;
  int INrow = blockIdx.y * blockDim.y + threadIdx.y;
    
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
    INs[ty][tx] = IN[INrow*nIn + tx + i];
    Ws[ty][tx] = W[Wrow*nIn + (i+ty)];
    VarPs[ty][tx] = VarP[Wrow*nIn + (i+ty)];
    VarMs[ty][tx] = VarM[Wrow*nIn + (i+ty)];
    __syncthreads();
    
    // compute element-size multiplication
    for(unsigned int j=0; j<BLOCK_SIZE; j++) {
      // multiplication
      T temp = INs[ty][j] * Ws[j][tx];
      // Variation modeling
      temp = (temp > 0)? temp + VarPs[j][tx] : temp + VarMs[j][tx];
      // Accumulation
      psum += temp;
      accumCount += 1;
      // digitize psum
      if(accumCount >= accumN) {
        // quantize psum
        psum = (accumN==1)? roundf(psum) : roundf(psum/2)*2; 
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
  if((INrow<nBatch) && (Wrow<nOut)) // shut down kernels that are not in the range
    OUT[INrow*nOut + Wrow] = ScalarConvert<AccumT, T>::to(output_temp);
}

#include "generic/CrossbarLinearWvar.cu"
#include "THCGenerateFloatTypes.h"