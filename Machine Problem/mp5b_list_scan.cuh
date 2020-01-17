// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int scanIndicate) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int start;
  int inputStride;
  int t = threadIdx.x;

  if (scanIndicate == 0){
    start = 2*blockIdx.x*blockDim.x + t;
    inputStride = blockDim.x;
  }
  else {
    start = 2 * blockDim.x * (t + 1) - 1;
    inputStride = 2 * blockDim.x * BLOCK_SIZE;
  }



  __shared__ float T[2*BLOCK_SIZE];
  int stride = 1;

  if (start < len){
    T[t] = input[start];
  }
  else {
    T[t] = 0;
  }
  if (start + inputStride < len){
    T[t + blockDim.x] = input[start + inputStride];
  }
  else {
    T[t + blockDim.x] = 0;
  }

  while (stride < 2*BLOCK_SIZE){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      T[index] += T[index - stride];
    }
    stride = stride*2;
  }
  stride = BLOCK_SIZE/2;
  while (stride > 0){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE){
      T[index+stride] += T[index];
    }
    stride = stride / 2;
  }

  __syncthreads();
  int point = 2*blockIdx.x*blockDim.x + t;

  if (point < len){
    output[point] = T[t];
  }
  if (point + blockDim.x < len){
    output[point + blockDim.x] = T[t + blockDim.x];
  }
}

__global__ void add(float *input, float *output, float *sum, int len) {

  int x = threadIdx.x + (blockIdx.x * blockDim.x * 2);

  __shared__ float addition;
  if (threadIdx.x == 0)
    addition = blockIdx.x == 0 ? 0 : sum[blockIdx.x - 1];

  __syncthreads();

  for(int i = 0; i < 2; i++){
    output[x + i * blockDim.x] = input[x + i * blockDim.x] + addition;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanStore;
  float *deviceScanSums;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanStore, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSums, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int DimGrid = ceil(numElements / float(2 * BLOCK_SIZE));
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, BLOCK_SIZE>>>(deviceInput, deviceScanStore, numElements, 0);
  cudaDeviceSynchronize();

  scan<<<1, BLOCK_SIZE>>>(deviceScanStore, deviceScanSums, numElements, 1);
  cudaDeviceSynchronize();

  add<<<DimGrid, BLOCK_SIZE>>>(deviceScanStore, deviceOutput, deviceScanSums, numElements);
  cudaDeviceSynchronize();


  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
