#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CrossbarCompute.cu"
#else

#include "../common.h"

void THNN_(CrossbarCompute_updateOutput)(
          THCState *state,
          THCTensor *output,
          THCTensor *input,
          THCTensor *weight,
          int accumN)
{
  THCUNN_assertSameGPU(state, 3, output, input, weight);
  
  // get parameters
  long nframe = 0;
  long nIn = 0;
  long nOut = 0;
  long nPsum = 0;
  
  int ndims = THCTensor_(nDimension)(state, input);
  
  if (ndims == 1) {
  }
  else if (ndims == 2) {
    nframe = THCTensor_(size)(state, input, 0);
    nIn = THCTensor_(size)(state, input, 1);
    nOut = THCTensor_(size)(state, weight, 0);
    nPsum = nIn / accumN;
    // resize output and make input continuous
    THCTensor_(resize3d)(state, output, nframe, nOut, nPsum);
    input = THCTensor_(newContiguous)(state, input);
    
    // set dimension of block and gird
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nOut+threads.x-1)/threads.x, (nframe+threads.y-1)/threads.y);
    
     
    // Execute the kernel
    cunn_CrossbarCompute_updateOutput_kernel<real, accreal><<<grid, threads>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCTensor_(data)(state, weight),
          accumN,
          nframe,
          nIn,
          nOut);
  }
  
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

