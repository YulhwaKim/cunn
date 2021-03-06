#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

#define BLOCK_SIZE 16

template <typename T, typename AccumT>
__global__ void cunn_CrossbarCompute_updateOutput_kernel(T *OUT, T *IN, T *W, int accumN, long nBatch, long nIn, long nOut)
{
  // index of output matrix
  int Wrow = blockIdx.x * blockDim.x + threadIdx.x;
  int INrow = blockIdx.y * blockDim.y + threadIdx.y;
  
  // y-dim of OUT
  long nY_OUT = nIn / accumN; // nIN should be divisible by accumN
  
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
    INs[ty][tx] = IN[INrow*nIn + tx + i];
//     Ws[ty][tx] = W[0];
    Ws[ty][tx] = W[Wrow*nIn + (i+ty)];
//     Ws[ty][tx] = W[(i+ty)*nOut + Wrow];
    __syncthreads();
    
    // compute element-size multiplication
    for(unsigned int j=0; j<BLOCK_SIZE; j++) {
      // do the accumulation
      temp += INs[ty][j] * Ws[j][tx];
      accumCount += 1;
      if(accumCount >= accumN) {
        // update outputs
        if((INrow<nBatch) && (OUTcol<nY_OUT) && (Wrow<nOut)) { // shut down kernels that are not in the range
          OUT[INrow*nY_OUT*nOut + Wrow*nY_OUT + OUTcol] = ScalarConvert<AccumT, T>::to(temp);
//           OUT[INrow*nY_OUT*nOut + OUTcol*nOut + Wrow] = ScalarConvert<AccumT, T>::to(temp);
        }
        // update or reset states
        OUTcol += 1;
        temp = 0;
        accumCount = 0;
      }
    }
    __syncthreads();
    i += BLOCK_SIZE;
  }
}

#include "generic/CrossbarCompute.cu"
#include "THCGenerateFloatTypes.h"
