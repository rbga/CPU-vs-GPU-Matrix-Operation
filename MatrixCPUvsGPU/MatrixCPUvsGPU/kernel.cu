
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate a random positive integer matrix
void generateRandomMatrix(int rows, int cols, int* matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = rand() % 100 + 1;  // Generate random positive integer between 1 and 100
        }
    }
}

// Function to print a matrix
void printMatrix(int rows, int cols, int* matrix)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
    //printf("\n");
}

// Matrix addition using CPU
void matrixAdditionCPU(int rows, int cols, int* matrixA, int* matrixB, int* result)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] = matrixA[i * cols + j] + matrixB[i * cols + j];
        }
    }
}

// Matrix subtraction using CPU
void matrixSubtractionCPU(int rows, int cols, int* matrixA, int* matrixB, int* result)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] = matrixA[i * cols + j] - matrixB[i * cols + j];
        }
    }
}

// Matrix multiplication using CPU
void matrixMultiplicationCPU(int rowsA, int colsA, int colsB, int* matrixA, int* matrixB, int* result)
{
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < colsB; j++)
        {
            int sum = 0;
            for (int k = 0; k < colsA; k++)
            {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}

// Matrix transpose using CPU
void matrixTransposeCPU(int rows, int cols, int* matrix, int* result)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

// Matrix addition using CUDA
__global__ void matrixAdditionGPU(int rows, int cols, int* matrixA, int* matrixB, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        result[i * cols + j] = matrixA[i * cols + j] + matrixB[i * cols + j];
    }
}

// Matrix subtraction using CUDA
__global__ void matrixSubtractionGPU(int rows, int cols, int* matrixA, int* matrixB, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        result[i * cols + j] = matrixA[i * cols + j] - matrixB[i * cols + j];
    }
}

// Matrix multiplication using CUDA
__global__ void matrixMultiplicationGPU(int rowsA, int colsA, int colsB, int* matrixA, int* matrixB, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rowsA && j < colsB)
    {
        int sum = 0;
        for (int k = 0; k < colsA; k++)
        {
            sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
        }
        result[i * colsB + j] = sum;
    }
}

// Matrix transpose using CUDA
__global__ void matrixTransposeGPU(int rows, int cols, int* matrix, int* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols)
    {
        result[j * rows + i] = matrix[i * cols + j];
    }
}

int main()
{
    int rows, cols;
    printf("Rows and Column must be equal, Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);

    if (rows != cols)
    {
        while (rows != cols)
        {
            printf("\n\nError! Rows and Columns must be equal");
            printf("\nEnter the number of rows: ");
            scanf("%d", &rows);
            printf("Enter the number of columns: ");
            scanf("%d", &cols);
            if (rows == cols)
                break;
        }
    }

    int size = rows * cols;
    size_t bytes = size * sizeof(int);

    // Allocate memory for matrices on CPU
    int* matrixA_CPU = (int*)malloc(bytes);
    int* matrixB_CPU = (int*)malloc(bytes);
    int* result_CPU = (int*)malloc(bytes);

    // Allocate memory for matrices on GPU
    int* matrixA_GPU;
    int* matrixB_GPU;
    int* result_GPU;
    cudaMalloc((void**)&matrixA_GPU, bytes);
    cudaMalloc((void**)&matrixB_GPU, bytes);
    cudaMalloc((void**)&result_GPU, bytes);

    // Generate random matrices
    generateRandomMatrix(rows, cols, matrixA_CPU);
    generateRandomMatrix(rows, cols, matrixB_CPU);

    // Print the matrices
    printf("\nMatrix A:\n");
    printMatrix(rows, cols, matrixA_CPU);
    printf("\nMatrix B:\n");
    printMatrix(rows, cols, matrixB_CPU);

    printf("\n------------------------------------------------------------------------\n");

    // Copy matrices from CPU to GPU
    cudaMemcpy(matrixA_GPU, matrixA_CPU, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_GPU, matrixB_CPU, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Perform matrix operations using CPU
    clock_t start_CPU, end_CPU;
    clock_t start_GPU, end_GPU;

    {
        // Matrix Addition - CPU
        start_CPU = clock();
        matrixAdditionCPU(rows, cols, matrixA_CPU, matrixB_CPU, result_CPU);
        end_CPU = clock();
        double cpuTime_Addition = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

        printf("\n\nMatrix Addition (CPU):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (CPU): %.6f seconds\n", cpuTime_Addition);
    }
    {
        // Matrix Addition - GPU (CUDA)
        start_GPU = clock();
        matrixAdditionGPU <<<numBlocks, threadsPerBlock >>> (rows, cols, matrixA_GPU, matrixB_GPU, result_GPU);
        cudaDeviceSynchronize();
        end_GPU = clock();
        double gpuTime_Addition = ((double)(end_GPU - start_GPU)) / CLOCKS_PER_SEC;

        // Copy the result matrix from GPU to CPU
        cudaMemcpy(result_CPU, result_GPU, bytes, cudaMemcpyDeviceToHost);

        printf("\nMatrix Addition (GPU - CUDA):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (GPU - CUDA): %.6f seconds\n", gpuTime_Addition);
    }
    
    printf("\n------------------------------------------------------------------------\n");

    {
        // Matrix Subtraction - CPU
        start_CPU = clock();
        matrixSubtractionCPU(rows, cols, matrixA_CPU, matrixB_CPU, result_CPU);
        end_CPU = clock();
        double cpuTime_Subtraction = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

        printf("\n\nMatrix Subtraction (CPU):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (CPU): %.6f seconds\n", cpuTime_Subtraction);
    }
    {
        // Matrix Subtraction - GPU (CUDA)
        start_GPU = clock();
        matrixSubtractionGPU <<<numBlocks, threadsPerBlock >>> (rows, cols, matrixA_GPU, matrixB_GPU, result_GPU);
        cudaDeviceSynchronize();
        end_GPU = clock();
        double gpuTime_Subtraction = ((double)(end_GPU - start_GPU)) / CLOCKS_PER_SEC;

        // Copy the result matrix from GPU to CPU
        cudaMemcpy(result_CPU, result_GPU, bytes, cudaMemcpyDeviceToHost);

        printf("\nMatrix Subtraction (GPU - CUDA):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (GPU - CUDA): %.6f seconds\n", gpuTime_Subtraction);
    }

    printf("\n------------------------------------------------------------------------\n");

    {
        // Matrix Multiplication - CPU
        start_CPU = clock();
        matrixMultiplicationCPU(rows, cols, cols, matrixA_CPU, matrixB_CPU, result_CPU);
        end_CPU = clock();
        double cpuTime_Multiplication = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

        printf("\n\nMatrix Multiplication (CPU):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (CPU): %.6f seconds\n", cpuTime_Multiplication);
    }
    {
        // Matrix Multiplication - GPU (CUDA)
        start_GPU = clock();
        matrixMultiplicationGPU <<<numBlocks, threadsPerBlock >>> (rows, cols, cols, matrixA_GPU, matrixB_GPU, result_GPU);
        cudaDeviceSynchronize();
        end_GPU = clock();
        double gpuTime_Multiplication = ((double)(end_GPU - start_GPU)) / CLOCKS_PER_SEC;

        // Copy the result matrix from GPU to CPU
        cudaMemcpy(result_CPU, result_GPU, bytes, cudaMemcpyDeviceToHost);

        printf("\nMatrix Multiplication (GPU - CUDA):\n");
        printMatrix(rows, cols, result_CPU);
        printf("Time taken (GPU - CUDA): %.6f seconds\n", gpuTime_Multiplication);
    }

    printf("\n------------------------------------------------------------------------\n");

    {
        // Matrix Transpose - CPU
        start_CPU = clock();
        matrixTransposeCPU(rows, cols, matrixA_CPU, result_CPU);
        end_CPU = clock();
        double cpuTime_Transpose = ((double)(end_CPU - start_CPU)) / CLOCKS_PER_SEC;

        printf("\n\nMatrix Transpose (CPU):\n");
        printMatrix(cols, rows, result_CPU);
        printf("Time taken (CPU): %.6f seconds\n", cpuTime_Transpose);
    }
    {
        // Matrix Transpose - GPU (CUDA)
        start_GPU = clock();
        matrixTransposeGPU <<<numBlocks, threadsPerBlock >>> (rows, cols, matrixA_GPU, result_GPU);
        cudaDeviceSynchronize();
        end_GPU = clock();
        double gpuTime_Transpose = ((double)(end_GPU - start_GPU)) / CLOCKS_PER_SEC;

        // Copy the result matrix from GPU to CPU
        cudaMemcpy(result_CPU, result_GPU, bytes, cudaMemcpyDeviceToHost);

        printf("\nMatrix Transpose (GPU - CUDA):\n");
        printMatrix(cols, rows, result_CPU);
        printf("Time taken (GPU - CUDA): %.6f seconds\n", gpuTime_Transpose);
    }

    printf("\n------------------------------------------------------------------------\n");
    
    // Free memory
    free(matrixA_CPU);
    free(matrixB_CPU);
    free(result_CPU);
    cudaFree(matrixA_GPU);
    cudaFree(matrixB_GPU);
    cudaFree(result_GPU);

    return 0;
}
