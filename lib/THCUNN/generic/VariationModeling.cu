#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/VariationModeling.cu"
#else

#include "../common.h"

void THNN_(VariationModeling_updateOutput)(
          THCState *state,
          THCTensor *output,
          THCTensor *input,
          THCTensor *ptable,
          int accumN)
          // THCTensor *) // ref is for debugging
{
//   THCUNN_assertSameGPU(state, 4, output, input, ptable, ref);
  THCUNN_assertSameGPU(state, 3, output, input, ptable);
  
  // get parameters
  int ndims = THCTensor_(nDimension)(state,input);
  long zdim = 1;
  long ydim = 1;
  long xdim = 1;
  if (ndims == 2) {
    zdim = 1;
    ydim = THCTensor_(size)(state, input, 0);
    xdim = THCTensor_(size)(state, input, 1);
  }
  if (ndims == 3) {
    zdim = THCTensor_(size)(state, input, 0);
    ydim = THCTensor_(size)(state, input, 1);
    xdim = THCTensor_(size)(state, input, 2);
  }
  long nRow = THCTensor_(size)(state, ptable, 0);
  long nCol = THCTensor_(size)(state, ptable, 1);
          
  // for debugging, print ptable
  real *temp = THCTensor_(data)(state, ptable);
  for(long i=0; i<nRow; i++) {
            for(long j=0; j<nCol; j++) {
                      printf("%.1f ", ptable[i*nCol+j]);
            }
            printf("\n");       
  }

  // resize output and make input continuous
  THCTensor_(resizeAs)(state, output, input);
  input = THCTensor_(newContiguous)(state, input);
  
   // check if BLOCK_SIZE is properly set
//   int check = BLOCK_SIZE;
//   printf("BLOCK_SIZE shoulbe be 32 and it is '%d'\n", check);
  
  // set dimension of block and grid
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((xdim + threads.x - 1)/threads.x, (ydim + threads.y - 1)/threads.y); 
          
  cunn_VariationModeling_updateOutput_kernel<real><<<grid, threads, nRow*nCol*sizeof(real)>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          xdim,
          ydim,
          zdim,
          THCTensor_(data)(state, ptable),
          nRow,
          nCol,
          accumN);
//           THCTensor_(data)(state, ref));
  
  // error checking
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
  {
    THError(cudaGetErrorString(errcode));
  }

  // free input
  THCTensor_(free)(state, input);
}


#endif

