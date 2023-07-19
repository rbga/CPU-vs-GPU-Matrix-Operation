# CPUvsGPUMatrixOperation

A performance comparison of standard matrix functions between CPU and GPU using Nvidia CUDA on Visual Studio using C++

## Code

The code begins by including the necessary CUDA runtime and device launch parameters headers, as well as standard C libraries for input/output and random number generation.

The main functions in the code are as follows:

* generateRandomMatrix: This function generates a random positive integer matrix of specified dimensions.
* printMatrix: This function prints a matrix.
* matrixAdditionCPU: This function performs matrix addition using the CPU.
* matrixSubtractionCPU: This function performs matrix subtraction using the CPU.
* matrixMultiplicationCPU: This function performs matrix multiplication using the CPU.
* matrixTransposeCPU: This function computes the transpose of a matrix using the CPU.
* matrixAdditionGPU: This CUDA kernel performs matrix addition using the GPU.
* matrixSubtractionGPU: This CUDA kernel performs matrix subtraction using the GPU.
* matrixMultiplicationGPU: This CUDA kernel performs matrix multiplication using the GPU.
* matrixTransposeGPU: This CUDA kernel computes the transpose of a matrix using the GPU.

In the main function, the code prompts the user to enter the number of rows and columns for the matrices. It handles the case where the rows and columns are not equal and prompts the user to enter them again.

The code then allocates memory for the matrices on both the CPU and GPU. It generates random matrices, prints them, and performs matrix operations using both the CPU and GPU implementations. The execution times for each operation are displayed.

Finally, the memory is freed, and the program terminates.

## Output

OUTPUT
------

Rows and Column must be equal, Enter the number of rows: 2
Enter the number of columns: 3

Error! Rows and Columns must be equal
Enter the number of rows: 5
Enter the number of columns: 5

Matrix A:
| 42 | 68 | 35 |  1 | 70 |
| 25 | 79 | 59 | 63 | 65 |
|  6 | 46 | 82 | 28 | 62 |
| 92 | 96 | 43 | 28 | 37 |
| 92 |  5 |  3 | 54 | 93 |

Matrix B:
| 83 | 22 |  17 | 19 | 96 |
| 48 | 27 |  72 | 39 | 70 |
| 13 | 68 | 100 | 36 | 95 |
|  4 | 12 |  23 | 34 | 74 |
| 65 | 42 |  12 | 54 | 69 |

------------------------------------------------------------------------

Matrix Addition (CPU):
| 125 |  90 |  52 |  20 | 166 |
|  73 | 106 | 131 | 102 | 135 |
|  19 | 114 | 182 |  64 | 157 |
|  96 | 108 |  66 |  62 | 111 |
| 157 |  47 |  15 | 108 | 162 |
Time taken (CPU): 0.000000 seconds

Matrix Addition (GPU - CUDA):
| 125 |  90 |  52 |  20 | 166 |
|  73 | 106 | 131 | 102 | 135 |
|  19 | 114 | 182 |  64 | 157 |
|  96 | 108 |  66 |  62 | 111 |
| 157 |  47 |  15 | 108 | 162 |
Time taken (GPU - CUDA): 0.000000 seconds

------------------------------------------------------------------------

Matrix Subtraction (CPU):
| -41 |  46 |  18 | -18 | -26 |
| -23 |  52 | -13 |  24 |  -5 |
|  -7 | -22 | -18 |  -8 | -33 |
|  88 |  84 |  20 |  -6 | -37 |
|  27 | -37 |  -9 |   0 |  24 |
Time taken (CPU): 0.000000 seconds

Matrix Subtraction (GPU - CUDA):
| -41 |  46 |  18 | -18 | -26 |
| -23 |  52 | -13 |  24 |  -5 |
|  -7 | -22 | -18 |  -8 | -33 |
|  88 |  84 |  20 |  -6 | -37 |
|  27 | -37 |  -9 |   0 |  24 |
Time taken (GPU - CUDA): 0.001000 seconds

------------------------------------------------------------------------

Matrix Multiplication (CPU):
| 11759 |  8092 |  9973 |  8524 | 17021 |
| 11111 | 10181 | 14242 | 11332 | 22682 |
|  7914 |  9890 | 13002 |  9160 | 17936 |
| 15320 |  9430 | 13864 |  9990 | 24262 |
| 14176 |  6917 |  4582 |  8909 | 19880 |
Time taken (CPU): 0.000000 seconds

Matrix Multiplication (GPU - CUDA):
| 11759 |  8092 |  9973 |  8524 | 17021 |
| 11111 | 10181 | 14242 | 11332 | 22682 |
|  7914 |  9890 | 13002 |  9160 | 17936 |
| 15320 |  9430 | 13864 |  9990 | 24262 |
| 14176 |  6917 |  4582 |  8909 | 19880 |
Time taken (GPU - CUDA): 0.000000 seconds

------------------------------------------------------------------------

Matrix Transpose (CPU):
| 42 | 25 |  6 | 92 | 92 |
| 68 | 79 | 46 | 96 |  5 |
| 35 | 59 | 82 | 43 |  3 |
|  1 | 63 | 28 | 28 | 54 |
| 70 | 65 | 62 | 37 | 93 |
Time taken (CPU): 0.000000 seconds

Matrix Transpose (GPU - CUDA):
| 42 | 25 |  6 | 92 | 92 |
| 68 | 79 | 46 | 96 |  5 |
| 35 | 59 | 82 | 43 |  3 |
|  1 | 63 | 28 | 28 | 54 |
| 70 | 65 | 62 | 37 | 93 |
Time taken (GPU - CUDA): 0.000000 seconds

------------------------------------------------------------------------
