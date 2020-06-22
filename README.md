# ECE408-Applied Paralle Programming-Machine Problem and Project

The aim of this course is to provide knowledge and hands-on experience in developing software for processors with massively parallel computing resources. Our students are required to program a massively parallel GPU system using CUDA, which is a popular commercial language extension of C/C++ for GPU programming. 

## Machine Problem Description 
### MP1 Vector Addition 
In this mp, I used CUDA API to implement a simple vector addition kernel. 

### MP2 Basic Matrix Multiplication 
The purpose of this lab is to implement a basic dense matrix multiplication routine.

### MP3 Tiled Matrix Multiplication
The purpose of this lab is to implement a tiled dense matrix multiplication routine using shared memory.

### MP4 3D convolution
The purpose of this lab is to implement a 3D convolution using constant memory for the kernel and 3D shared memory tiling.

### MP5.1 List Reduction 
Implement a kernel and associated host code that performs reduction of a 1D list stored in a C array. The reduction should give the sum of the list. The kernel can be able to handle input lists of arbitrary length.

### MP5.2 List Scan
The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used will be addition. You should implement the work- efficient kernel discussed in lecture. Your kernel should be able to handle input lists of arbitrary length. However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

### Histogram Equalization 
The purpose of this lab is to implement an efficient histogramming equalization algorithm for an input image. Like the image convolution MP, the image is represented as RGB float values. You will convert that to GrayScale unsigned char values and compute the histogram. Based on the histogram, you will compute a histogram equalization function which you will then apply to the original image to get the color corrected image.

### Sparse Matrix Multiplication 
The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based on the Jagged Diagonal Storage (JDS) transposed format.
