#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/CrossbarSpatialConvoluion.cu"
#else

static inline void THNN_(CrossbarSpatialConvolution_shapecheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *weight,
                         int kH, int kW, int dH, int dW, int padH, int padW){
  THArgCheck(kW > 0 && kH > 0, 9,
            "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
            "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THArgCheck(state, weight->nDimension == 2 || weight->nDimension == 4, 5, weight,
            "2D or 4D weight tensor expected, but got: %s");
  
  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;
  
  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }
  
  THCUNN_argcheck(state, ndim == 3 || ndim == 4, 2, input,
                 "3D or 4D input tensor expected but got: %s");
  
  long nInputPlane = weight->size[1] / (kH * kW);
  long inputHeight = input->size[dimh];
  long inputWidth = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth = (inputWidth + 2*padW - kW) / dW + 1;
  
  if (outputWidth < 1 || outputHeight < 1)
    THError("Given input size: (%d x %d x %d). "
            "Calculated output size: (%d x %d x %d). Output size is too small", 
             nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight, outputWidth);
  
  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane); 
}

void THNN_(CrossbarSpatialConvolution_updateoutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *columns,
           int accumN,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {
  
  THCUNN_assertSameGPU(state, 4, input, output, weight, columns);
  THArgCheck(THCTensor_(isContiguous)(state, weight), 4,
             "weight tensor has to be contiguous");
  
  // convert 4D weight into 2D weight
  int freeWeight = 0;
  if (weight->nDimension == 4) {
    long s1 = weight->size[0];
    long s2 = weight->size[1] * weight->size[2] * weight->size[3];
    weight = THCTensor_(newWithStorage2d)(state, weight->storage, weight->storageOffset, s1, -1, s2, -1);
    freeWeight = 1;
  }
  
  THNN_(CrossbarSpatialConvolution_shapeCheck)
    (sate, input, weight, kH, kW, dH, dW, padH, padW);
  
  // make input contiguous and 4D
  input = THCTensor_(newContiguous)(state, input);
  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }
  
  // Params:
  long nInputPlane = weight->size[1]/(kH*kW);
  long nIn = weight->size[1];
  long nOutputPlane = weight->size[0];
  long inputWidth = input->size[3];
  long inputHeight = input->size[2];
  long outputWidth = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long nOutSpatial = outputWidth * outputHeight;
  long batchSize = input->size[0];
  long nPsum = weight->size[1] / accumN;
  //Check if nPsum is valid
  THArgCheck(nPsum > 0 && weight->size[1] == nPsum * accumN, 101,
            "Number of input per convolution should be divisible by accumN, but we got number of input: %ld, accumN: %d, nPsum: %ld",
             weight->size[1], accumN, nPsum);
  
  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, outputHeight, outputWidth, nPsum);
  
  // Resize temporary columns
  THCTensor_(resize2d)(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth);
  
  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *output_n = THCTensor_(new)(state);
  
  // set dimension of block and grid
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((nOutputPlane+threads.x-1)/threads.x, (nOutSpatial+threads.y-1)/threads.y);
  
  // For each elt in batch, do:
  for (long elt = 0; elt < batchSize; elt ++) {
    // Matrix multiply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, output_n, output, 0, elt);
    
    // Extract columns:
    im2col(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input_n),
      nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      1, 1, THCTensor_(data)(state, columns)
    );
    
    // Execute the kernel
    cunn_CrossbarSpatialConvolution_updateOutput_frame_kernel<real, accreal><<<grid, threads>>>(
          THCTensor_(data)(state, output_n),
          THCTensor_(data)(state, columns),
          THCTensor_(data)(state, weight),
          accumN,
          nIn,
          nOutSpatial,
          nOutputPlane,
          nPsum);
  }
  
  // free memorys
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, output_n);
  if (freeWeight)
    THCTensor_(free)(state, weight);
  
  // Resize output
  if (batch == 0) {
    THCTensor_(resize3d)(state, output, nOutputPlane, outputHeight, outputWidth, nPsum);
    THCTensor_(resize3d)(state, input, nInputPlane, inputHeight, inputWidth);
  }
  THCTensor_(free)(state, input);
  
}


#endif
