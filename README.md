# CPUvsGPUMatrixOperation
A performance comparison of standard matrix functions between CPU and GPU using Nvidia CUDA on Visual Studio using C++

This code is a CUDA program that performs various matrix operations including addition, subtraction, multiplication, and transpose. The program provides both CPU and GPU implementations for these operations and compares their execution times.

The code begins by including the necessary CUDA runtime and device launch parameters headers, as well as standard C libraries for input/output and random number generation.

The main functions in the code are as follows:

1. generateRandomMatrix: This function generates a random positive integer matrix of specified dimensions.
2. printMatrix: This function prints a matrix.
3. matrixAdditionCPU: This function performs matrix addition using the CPU.
4. matrixSubtractionCPU: This function performs matrix subtraction using the CPU.
5. matrixMultiplicationCPU: This function performs matrix multiplication using the CPU.
6. matrixTransposeCPU: This function computes the transpose of a matrix using the CPU.
7. matrixAdditionGPU: This CUDA kernel performs matrix addition using the GPU.
8. matrixSubtractionGPU: This CUDA kernel performs matrix subtraction using the GPU.
9. matrixMultiplicationGPU: This CUDA kernel performs matrix multiplication using the GPU.
10. matrixTransposeGPU: This CUDA kernel computes the transpose of a matrix using the GPU.

In the main function, the code prompts the user to enter the number of rows and columns for the matrices. It handles the case where the rows and columns are not equal and prompts the user to enter them again.

The code then allocates memory for the matrices on both the CPU and GPU. It generates random matrices, prints them, and performs matrix operations using both the CPU and GPU implementations. The execution times for each operation are displayed.

Finally, the memory is freed, and the program terminates.

This code can be used as a starting point for CUDA programming and benchmarking matrix operations on both the CPU and GPU.
