// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32 //@@ You can change this

//@@ insert code here
__global__ void float2uc(float *input, unsigned char *output, int width, int height ){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = blockIdx.z*height*width + ty*width +tx;
    output[idx] = (unsigned char) (255*input[idx]);
  }
}

__global__ void rgb2gray(unsigned char *input, unsigned char *output, int width, int height){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = ty*width + tx;
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx + 1];
    unsigned char b = input[3*idx + 2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void gray2histogram (unsigned char *input, unsigned int *histo, int width, int height){

  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  int t = threadIdx.x + threadIdx.y*blockDim.x;

  if (t < 256){
    histo_private[t] = 0;
  }

  __syncthreads();

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = ty*width + tx;
    atomicAdd( &(histo_private[input[idx]]), 1);
  }

  __syncthreads();

  if (t < 256){
    atomicAdd( &(histo[t]),histo_private[t]);
  }
}

__global__ void histo2cdf(unsigned int *input, float *output, int width, int height){

  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int t = threadIdx.x + blockDim.x*blockIdx.x;
  // cdf [threadIdx.x] = input[threadIdx.x];
  if (t < HISTOGRAM_LENGTH){
    cdf[t] = input[t];
  }

  unsigned int stride = 1;
  while (stride <= HISTOGRAM_LENGTH/2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < HISTOGRAM_LENGTH ){
      cdf[index] += cdf[index - stride];
    }
    stride = stride * 2;
  }

  stride = HISTOGRAM_LENGTH/4;
  while (stride > 0){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < HISTOGRAM_LENGTH){
      cdf[index+stride] += cdf[index];
    }
    stride = stride / 2;
  }
  __syncthreads();
  if (t < HISTOGRAM_LENGTH){
    output[t] = cdf[t]/((float)(width*height));
  }
  // output[threadIdx.x] = cdf[threadIdx.x]/(float)(width*height);
}

__global__ void equalization(unsigned char *image, float *cdf, int width, int height){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
     int idx = blockIdx.z * (width * height) + ty * (width) + tx;

     float correct_color = 255*(cdf[image[idx]] - cdf[0])/(1.0 - cdf[0]);
     float clamp = min(max(correct_color, 0.0), 255.0);
     image[idx] = (unsigned char) (clamp);
  }
}

__global__ void uc2float(unsigned char *input, float *output, int width, int height){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = blockIdx.z*height*width + ty*width + tx;
    output[idx] = (float) (input[idx]/255.0);
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  //@@ Insert more code here
  float *deviceInput;
  unsigned char *deviceUC;
  unsigned char *deviceUCGray;
  unsigned int *deviceHistog;
  float *deviceCDF;
  float *deviceOutput;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  cudaMalloc((void**) &deviceInput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceUC, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceUCGray, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceHistog, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));

  cudaMemset((void *) deviceHistog, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(deviceInput, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);


  //@@ insert code here

  dim3 DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),imageChannels);
  dim3 DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  float2uc<<<DimGrid, DimBlock>>> (deviceInput,deviceUC,imageWidth,imageHeight);
  cudaDeviceSynchronize();

  DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),1);
  DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  rgb2gray<<<DimGrid,DimBlock>>>(deviceUC,deviceUCGray,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  gray2histogram<<<DimGrid,DimBlock>>>(deviceUCGray,deviceHistog,imageWidth,imageHeight);
  cudaDeviceSynchronize();

  DimGrid  = dim3(1, 1, 1);
  DimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  histo2cdf<<<DimGrid,DimBlock>>>(deviceHistog,deviceCDF,imageWidth,imageHeight);
  cudaDeviceSynchronize();

  DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),imageChannels);
  DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  equalization<<<DimGrid,DimBlock>>>(deviceUC,deviceCDF,imageWidth,imageHeight);
  cudaDeviceSynchronize();
  uc2float<<<DimGrid,DimBlock>>>(deviceUC,deviceOutput,imageWidth,imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceOutput, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceInput);
  free(hostInputImageData);
  free(hostOutputImageData);


  return 0;
}
