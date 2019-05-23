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
  
  // size of IN & W
  long size_IN = nBatch * nIn;
  long size_W = nIn * nOut;
    
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
  long IN_idx = 0;
  long W_idx = 0;
  
  while(i < nIn){
    // copy the data from global memory to shared memory
    IN_idx = INrow*nIn + tx + i;
    W_idx = Wrow*nIn + (i+ty);
    INs[ty][tx] = (IN_idx < size_IN)? IN[IN_idx] : IN[size_IN - 1];
    Ws[ty][tx] = (W_idx < size_W)? W[W_idx] : W[size_W - 1];
    VarPs[ty][tx] = (W_idx < size_W)? VarP[W_idx] : VarP[size_W - 1];
    VarMs[ty][tx] = (W_idx < size_W)? VarM[W_idx] : VarM[size_w - 1];
//     INs[ty][tx] = IN[INrow*nIn + tx + i];
//     Ws[ty][tx] = W[Wrow*nIn + (i+ty)];
//     VarPs[ty][tx] = VarP[Wrow*nIn + (i+ty)];
//     VarMs[ty][tx] = VarM[Wrow*nIn + (i+ty)];
    __syncthreads();
    
    // compute element-size multiplication
    for(unsigned int j=0; j<BLOCK_SIZE; j++) {
      if (i + j >= nIn) // finish accumulation on the end point of the matrix
        break;
      // multiplication
      T temp = INs[ty][j] * Ws[j][tx];
      // Variation modeling
      temp = (temp >= 0)? temp + VarPs[j][tx] : temp + VarMs[j][tx];
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
  if((INrow<nBatch) && (Wrow<nOut)) // shut down kernels that are not in the range
    OUT[INrow*nOut + Wrow] = ScalarConvert<AccumT, T>::to(output_temp);
}

#include "generic/CrossbarLinearWvar.cu"
#include "THCGenerateFloatTypes.h"
