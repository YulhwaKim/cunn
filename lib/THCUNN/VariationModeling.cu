#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

// libraries for random number generation
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 32

template <typename T>
__global__ void cunn_VariationModeling_updateOutput_kernel(
  T *OUT, T *IN, long xdim, long ydim, long zdim, T *PTABLE, long nRow, long nCol, int accumN, long long seed)//, T *REF) // REF is for debugging
{
  // index of data 
  int INcol = blockIdx.x * blockDim.x + threadIdx.x;
  int INrow = blockIdx.y * blockDim.y + threadIdx.y;
  
  // initialize curand
  curandState = s;
  curand_init(seed, INcol, INrow, &s);
  
  // thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // transitionWindow
  long transitionWindow = (nCol-1)/2;
  
  // dynamic shared memory allocation for PTABLE
//   extern __shared__ T PTABLEs [];
  SharedMem<T> smem;
  T *PTABLEs = smem.getPointer();

  
  // move PTABLE into shared memory
  int col_iter = (nCol + blockDim.x - 1) / blockDim.x;
  int row_iter = (nRow + blockDim.y - 1) / blockDim.y;
  for(unsigned int i=0; i<row_iter; i++) {
   for(unsigned int j=0; j<col_iter; j++) {
     int xIdx = j*blockDim.x + tx;
     int yIdx = i*blockDim.y + ty;
     if((xIdx < nCol) && (yIdx < nRow)) {
       PTABLEs[yIdx*nCol + xIdx] = PTABLE[yIdx*nCol + xIdx];
     }
   }
  }
  __syncthreads();
  
  if((INcol >= xdim) || (INrow >= ydim)) {
    return ;
  }
  
  // each thread models variation on given 2D matrix
  // thus, have to repeat on z-dim elements
  srand(time(NULL));
  for(long i=0; i<zdim; i++) {
   // STEP1. get data and row index of probability table
    long INidx = i*xdim*ydim + INrow*xdim + INcol;
    int value = ScalarConvert<T, int>::to(IN[INidx]);
    int rowIdx = (value + accumN) / 2;
    // STEP2. generate reference point
//     T refpoint = REF[INidx];
    T refpoint = ScalarConvert<float, T>::to(curand_uniform(s));
    // STEP3. find the column index of probability table and change the data
    for(int j=0; j<nCol; j++) {
      T prob = PTABLEs[rowIdx*nCol + j];
      if(((prob > 0) && (prob > refpoint)) || (j==nCol-1)) {
        OUT[INidx] = ScalarConvert<int, T>::to(value + 2*(j - transitionWindow));
        break;
      }
    }
    __syncthreads();
  }
  
}

#include "generic/VariationModeling.cu"
#include "THCGenerateFloatTypes.h"
