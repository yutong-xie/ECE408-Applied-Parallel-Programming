
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 16

#include <mxnet/base.h>
#include <stdio.h>
#include <math.h>

namespace mxnet
{
namespace op
{

//__constant__ float mc[12000];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int n = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH  + threadIdx.x;

    if ((w < (W_out)) && (h < (H_out))) {
      float acc=0;
      for(int c=0;c<C;c++) {
        acc+=x4d(n,c,h,w+0)*k4d(m,c,0,0)
            +x4d(n,c,h,w+1)*k4d(m,c,0,1)
            +x4d(n,c,h,w+2)*k4d(m,c,0,2)
            +x4d(n,c,h,w+3)*k4d(m,c,0,3)
            +x4d(n,c,h,w+4)*k4d(m,c,0,4)
            +x4d(n,c,h+1,w+0)*k4d(m,c,1,0)
            +x4d(n,c,h+1,w+1)*k4d(m,c,1,1)
            +x4d(n,c,h+1,w+2)*k4d(m,c,1,2)
            +x4d(n,c,h+1,w+3)*k4d(m,c,1,3)
            +x4d(n,c,h+1,w+4)*k4d(m,c,1,4)
            +x4d(n,c,h+2,w+0)*k4d(m,c,2,0)
            +x4d(n,c,h+2,w+1)*k4d(m,c,2,1)
            +x4d(n,c,h+2,w+2)*k4d(m,c,2,2)
            +x4d(n,c,h+2,w+3)*k4d(m,c,2,3)
            +x4d(n,c,h+2,w+4)*k4d(m,c,2,4)
            +x4d(n,c,h+3,w+0)*k4d(m,c,3,0)
            +x4d(n,c,h+3,w+1)*k4d(m,c,3,1)
            +x4d(n,c,h+3,w+2)*k4d(m,c,3,2)
            +x4d(n,c,h+3,w+3)*k4d(m,c,3,3)
            +x4d(n,c,h+3,w+4)*k4d(m,c,3,4)
            +x4d(n,c,h+4,w+0)*k4d(m,c,4,0)
            +x4d(n,c,h+4,w+1)*k4d(m,c,4,1)
            +x4d(n,c,h+4,w+2)*k4d(m,c,4,2)
            +x4d(n,c,h+4,w+3)*k4d(m,c,4,3)
            +x4d(n,c,h+4,w+4)*k4d(m,c,4,4);
      }
      y4d(n,m,h,w)=acc;
    }
    //y4d(blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y)=sum;
    #undef y4d
    #undef x4d
    #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    const int B = x.shape_[0]; //batches
    const int M = y.shape_[1]; //output channels
    const int C = x.shape_[1]; //input channels
    const int H = x.shape_[2]; //height of input
    const int W = x.shape_[3]; //width of input
    const int K = w.shape_[3]; //height and width of weights
    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil(W_out/(TILE_WIDTH * 1.0)); // number of horizontal tiles per output map
    int H_grid = ceil(H_out/(TILE_WIDTH * 1.0)); // number of vertical tiles per output map
    int Z = H_grid * W_grid;

    //cudaMemcpyToSymbol(mc, w.dptr_, sizeof(float) * M * C * K * K);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K,W_grid);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
