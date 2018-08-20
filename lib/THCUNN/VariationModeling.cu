#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "SharedMem.cuh"

// libraries for random number generation
#include <curand_kernel.h>

#define BLOCK_SIZE 32

template <typename T>
__global__ void cunn_VariationModeling_updateOutput_kernel(
  T *OUT, T *IN, long xdim, long ydim, long zdim, T *PTABLE, long nRow, long nCol, int accumN)//, T *REF) // REF is for debugging
{
  // index of data 
  int INcol = blockIdx.x * blockDim.x + threadIdx.x;
  int INrow = blockIdx.y * blockDim.y + threadIdx.y;
  
  // initialize curand
  curandState s;
  curand_init(clock64(), INcol, 0, &s);
  
  // thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // transitionWindow
  long transitionWindow = (nCol-1)/2;
  
  // dynamic shared memory allocation for PTABLE
//   extern __shared__ T PTABLEs [];
  SharedMem<T> smem;
  T *PTABLEs = smem.getPointer();

//  printf("nRow: %ld, nCol: %ld\n", nRow, nCol); //correct
  
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
  for(long i=0; i<zdim; i++) {
   // STEP1. get data and row index of probability table
    long INidx = i*xdim*ydim + INrow*xdim + INcol;
    int value = ScalarConvert<T, int>::to(IN[INidx]);
    int rowIdx = (value + accumN) / 2;
    // STEP2. generate reference point
//     T refpoint = REF[INidx];
    T refpoint = ScalarConvert<float, T>::to(curand_uniform(&s));
    // STEP3. find the column index of probability table and change the data
    for(int j=0; j<nCol; j++) {
//       T prob = PTABLEs[rowIdx*nCol + j];
      T prob = PTABLE[j*nRow + rowIdx];
//       printf("rowIdx: %d, colIdx: %d, prob: %.2f\n", rowIdx, j, prob);
      if(((prob > 0) && (prob > refpoint)) || (j==nCol-1)) {
        // printf("transitionWindow: %ld , value: %d, rowIdx: %d, refpoint: %.1f, j: %d\n", transitionWindow, value, rowIdx, refpoint, j);
        OUT[INidx] = ScalarConvert<int, T>::to(value + 2*(j - transitionWindow));
        printf("value: %d, refpoint: %.2f, output: %.1f, table row: %d, table col: %d, prob: %.2f\n", 
               value, refpoint, OUT[INidx], rowIdx, j, prob);
        break;
      }
    }
    __syncthreads();
  }
  
}

#include "generic/VariationModeling.cu"
#include "THCGenerateFloatTypes.h"
