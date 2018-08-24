#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CrossbarLinearWvar.cu"
#else

#include "../common.h"

void THNN_(CrossbarLinearWvar_updateOutput)(
          THCState *state,
          THCTensor *output,
          THCTensor *input,
          THCTensor *weight,
          THCTensor *VarP,
          THCTensor *VarM,
          int accumN)
{
  THCUNN_assertSameGPU(state, 5, output, input, weight, VarP, VarM);
  
  // get parameters
  long nframe = 0;
  long nIn = 0;
  long nOut = 0;
          
  // check if BLOCK_SIZE is properly set
//   int check = BLOCK_SIZE;
//   printf("BLOCK_SIZE shoulbe be 16 and it is '%d'\n", check);
  
  int ndims = THCTensor_(nDimension)(state, input);
  
  if (ndims == 1) {
  }
  else if (ndims == 2) {
    nframe = THCTensor_(size)(state, input, 0);
    nIn = THCTensor_(size)(state, input, 1);
    nOut = THCTensor_(size)(state, weight, 0);
    // resize output and make input continuous
    THCTensor_(resize2d)(state, output, nframe, nOut);
    input = THCTensor_(newContiguous)(state, input);
    
    // set dimension of block and gird
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nOut+threads.x-1)/threads.x, (nframe+threads.y-1)/threads.y);
     
    // Execute the kernel
    cunn_CrossbarLinearWvar_updateOutput_kernel<real, accreal><<<grid, threads>>>(
          THCTensor_(data)(state, output),
          THCTensor_(data)(state, input),
          THCTensor_(data)(state, weight),
          THCTensor_(data)(state, VarP),
          THCTensor_(data)(state, VarM),
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
