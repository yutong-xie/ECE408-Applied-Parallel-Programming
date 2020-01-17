
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

__global__ void convlayerforward(float *y, float *x, float *w, int B, int C, int H, int K, int W, int M)
{
    __shared__ float MaskTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float InputTile[TILE_WIDTH][TILE_WIDTH];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define k4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int column = blockIdx.x * TILE_WIDTH + tx;
    int unrollColumn = C*K*K;

    float acc = 0.0;
    int num_iterations = ceil(unrollColumn/(1.0*TILE_WIDTH));

    for (int i = 0; i < num_iterations; i++) {
      int lx = i*TILE_WIDTH + tx;
      int ly = i*TILE_WIDTH + ty;

      MaskTile[ty][tx] = 0;
      InputTile[ty][tx] = 0;

      int W_m = row;
      int W_c = lx/(K*K);
      int W_h = (lx%(K*K))/K;
      int W_w = (lx%(K*K))%K;

      if ((lx < unrollColumn) && (row < M)){
        MaskTile[ty][tx] = k4d(W_m, W_c, W_h, W_w);
      }
      else{
        MaskTile[ty][tx] = 0;
      }

      int X_b = b;
      int X_c = ly/(K*K);
      int X_p = (ly%(K*K))/K;
      int X_q = (ly%(K*K))%K;
      int X_h = column/W_out;
      int X_w = column%W_out;

      if (ly < unrollColumn && column < H_out*W_out){
        InputTile[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
      }
      else{
        InputTile[ty][tx] = 0;
      }
      __syncthreads();

      for (int q = 0; q < TILE_WIDTH; q++){
        acc += MaskTile[ty][q] * InputTile[q][tx];
      }

      // acc += MaskTile[ty][0] * InputTile[0][tx]
      //      + MaskTile[ty][1] * InputTile[1][tx]
      //      + MaskTile[ty][2] * InputTile[2][tx]
      //      + MaskTile[ty][3] * InputTile[3][tx]
      //      + MaskTile[ty][4] * InputTile[4][tx]
      //      + MaskTile[ty][5] * InputTile[5][tx]
      //      + MaskTile[ty][6] * InputTile[6][tx]
      //      + MaskTile[ty][7] * InputTile[7][tx]
      //      + MaskTile[ty][8] * InputTile[8][tx]
      //      + MaskTile[ty][9] * InputTile[9][tx]
      //      + MaskTile[ty][10] * InputTile[10][tx]
      //      + MaskTile[ty][11] * InputTile[11][tx]
      //      + MaskTile[ty][12] * InputTile[12][tx]
      //      + MaskTile[ty][13] * InputTile[13][tx]
      //      + MaskTile[ty][14] * InputTile[14][tx]
      //      + MaskTile[ty][15] * InputTile[15][tx];

      __syncthreads();
    }
    int Y_b = b;
    int Y_m = row;
    int Y_h = column / W_out;
    int Y_w = column % W_out;

    if (row < M && column < W_out*H_out)
      y4d(Y_b, Y_m, Y_h, Y_w) = acc;
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
    // int W_grid = ceil(W_out/(TILE_WIDTH * 1.0)); // number of horizontal tiles per output map
    // int H_grid = ceil(H_out/(TILE_WIDTH * 1.0)); // number of vertical tiles per output map
    // int Z = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);
    // Call the kernel
    // printf("M,C,H,K,W is : %d,%d,%d,%d,%d",M,C,H,K,W);
    convlayerforward<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_,B,C,H,K,W,M);
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
