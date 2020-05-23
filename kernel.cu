#include <cstdlib>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <windows.h>

#pragma comment(lib, "cudart") 

#define MAX_BYTES_CPU 214748347//*sizeof(short)
#define MAX_BYTES_GPU 885850113//*sizeof(short)

#define COUNT_OF_ROWS 
#define COUNT_OF_COLS 128
#define PARTS 2

#define BLOCK_SIZE 4

using namespace std;

__global__ void kernel(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix);
__global__ void optomizedKernel(int* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, int* resultMatrix,size_t pitch);
short** splitInSubmatrices(short* matrix);
void transformMatrixCPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol,short* resultMatrix);
void transformMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol,short* resultMatrix);
void transform2DMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix);
void transformBigMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix);
short* initializeMatrix(int matrixRows,int matrixCols);
void showMatrix(short* matrix, int matrixRows, int matrixCols);
void randomElements(short* matrix,int matrixRows, int matrixCols);
bool compareMatrix(short* matrix1, short* matrix2, int matrixRow, int magrixCol);
void checkOnError(cudaError_t cudaStatus);
void connectMatrices(short** arrayOfMatrices, short* matrix);

int main() {
	short* sourse = initializeMatrix(COUNT_OF_ROWS, COUNT_OF_COLS);
	short* result1 = initializeMatrix(COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2);
	short* result2 = initializeMatrix(COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2);
	short* mx = initializeMatrix(COUNT_OF_ROWS / PARTS, COUNT_OF_COLS * 2);
	randomElements(sourse, COUNT_OF_ROWS, COUNT_OF_COLS);
	/*int x1 = 5;
	int x2 = 6;
	int x6  = (x1 << 16) | x2;
	int x7 = x6 >> 16;
	cout << x7;*/	//transformMatrixCPU(sourse, COUNT_OF_ROWS, COUNT_OF_COLS, result1);
	//showMatrix(sourse, COUNT_OF_ROWS, COUNT_OF_COLS);
	cout << endl; 
	transformMatrixGPU(sourse, COUNT_OF_ROWS, COUNT_OF_COLS, result2);
	//unsigned short leastSignificantWord = 24;
	//unsigned short  mostSignificantWord = 13;
	//unsigned int  i = (unsigned int)mostSignificantWord << 16 | leastSignificantWord;
	//unsigned short  g = i >> 16;
	//showMatrix(result2, COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2);
	transform2DMatrixGPU(sourse, COUNT_OF_ROWS, COUNT_OF_COLS, result1);
	//showMatrix(result1, COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2);
    //transformBigMatrixGPU(sourse, COUNT_OF_ROWS, COUNT_OF_COLS, result2);
    //showMatrix(result2, COUNT_OF_ROWS/2, COUNT_OF_COLS*2);
	if (compareMatrix(result1, result2, COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2))
		cout << "OK" << endl; else cout << "MATRIX NOT EQUAL" << endl;
	//showMatrix(result2, COUNT_OF_ROWS / 2, COUNT_OF_COLS * 2);
	system("pause");
	cudaDeviceReset();
	return 0;
}

__device__ int extractShort(int x,int bytePos){
	if (bytePos == 0)
		return x & 0xFFFF;
	else return x >> 16;	
}

__device__ int makeInt(int x1, int x2) {
	return (x1 << 16) | x2;
}


__global__ void optomizedKernel(int* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, int* resultMatrix, size_t pitch) {
	int   colsId =  blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int   rowsId =  blockIdx.y * BLOCK_SIZE + threadIdx.y;

	__shared__  int sharedForInElements[BLOCK_SIZE*2][BLOCK_SIZE*2];
	__shared__  int sharedForOutElemens[BLOCK_SIZE*2][BLOCK_SIZE*2];

	int resultMatrixCol = colsId *2;
	int resultMatrixRow = rowsId/2;

	sharedForInElements[threadIdx.y][threadIdx.x] = 
		(((int*)((char*)sourseMatrix + rowsId * pitch))[colsId]);
	sharedForInElements[threadIdx.y+ BLOCK_SIZE][threadIdx.x] =
		(((int*)((char*)sourseMatrix + (rowsId + BLOCK_SIZE) * pitch))[colsId]);

    __syncthreads();

	int x1 = makeInt(extractShort(sharedForInElements[threadIdx.y + 1][threadIdx.x], 1),
		extractShort(sharedForInElements[threadIdx.y][threadIdx.x], 1));
	int x2 = makeInt(extractShort(sharedForInElements[threadIdx.y + 1][threadIdx.x], 0),
		extractShort(sharedForInElements[threadIdx.y][threadIdx.x], 0));

	sharedForInElements[threadIdx.y/2][threadIdx.x + BLOCK_SIZE] = 2;
	sharedForInElements[threadIdx.y/2][threadIdx.x] = 1;
	     
	__syncthreads();

	int* res = (int*)((char*)resultMatrix + resultMatrixRow * pitch);
	res[resultMatrixCol] = sharedForInElements[threadIdx.y/2][threadIdx.x];
	res[resultMatrixCol+1] = sharedForInElements[threadIdx.y/2][threadIdx.x + BLOCK_SIZE];
}

void transform2DMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix) {
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	short* sourseMatrixGPU;
	short* resultMatrixGPU;
	float resultTime;
	size_t pitch;
	checkOnError(cudaMallocPitch(&sourseMatrixGPU, &pitch, sourseMatrixCol * sizeof(short), sourseMatrixRow));
	checkOnError(cudaMemcpy2D(sourseMatrixGPU, pitch, sourseMatrix, sourseMatrixCol * sizeof(short), sourseMatrixCol * sizeof(short),
		sourseMatrixRow, cudaMemcpyHostToDevice));
	checkOnError(cudaMallocPitch(&resultMatrixGPU, &pitch, (sourseMatrixCol*2) * sizeof(short), sourseMatrixRow/2));
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(sourseMatrixCol / block.x, sourseMatrixRow / block.y);
	if (sourseMatrixCol % block.x != 0) grid.x++;
	if (sourseMatrixRow % block.y != 0) grid.y++;
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime);
	optomizedKernel << <grid, block >> > ((int*)sourseMatrixGPU, sourseMatrixRow, sourseMatrixCol,(int*)resultMatrixGPU, pitch);
	cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&resultTime, startTime, stopTime);
	cout << "Shared GPU time: " << resultTime << " ms" << endl;
	checkOnError(cudaMemcpy2D(resultMatrix, (sourseMatrixCol*2) * sizeof(short), resultMatrixGPU, pitch, (sourseMatrixCol*2) * sizeof(short), sourseMatrixRow/2,
		cudaMemcpyDeviceToHost));
	cudaFree(sourseMatrixGPU);
	cudaFree(resultMatrixGPU);
}

void transformMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol,short* resultMatrix) {
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	short* sourseMatrixGPU;
	short* resultMatrixGPU;
	float resultTime;
	checkOnError(cudaMalloc((void**)&sourseMatrixGPU, sourseMatrixRow  * sourseMatrixCol * sizeof(short)));
	checkOnError(cudaMemcpy(sourseMatrixGPU, sourseMatrix, sourseMatrixRow * sourseMatrixCol * sizeof(short), cudaMemcpyHostToDevice));
	checkOnError(cudaMalloc((void**)&resultMatrixGPU, sourseMatrixRow * sourseMatrixCol *  sizeof(short)));
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(sourseMatrixCol / block.x, sourseMatrixRow / block.y);
	if (sourseMatrixCol % block.x != 0) grid.x ++;
	if (sourseMatrixRow % block.y != 0) grid.y ++;
	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime);
	//for (int i = 0; i < 20; i++) {
		kernel << <grid, block >> > (sourseMatrixGPU, sourseMatrixRow, sourseMatrixCol, resultMatrixGPU);
	//}
	cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);
	cudaEventElapsedTime(&resultTime, startTime, stopTime);
	cout << "GPU time: " << resultTime << " ms" << endl;
	checkOnError(cudaMemcpy(resultMatrix, resultMatrixGPU, sourseMatrixRow  * sourseMatrixCol * sizeof(short),
		cudaMemcpyDeviceToHost));
	cudaFree(sourseMatrixGPU);
	cudaFree(resultMatrixGPU);
}

short** splitInSubmatrices(short* matrix) {
	int subMatrixRow=0,step = 0;
	short** Submatrices = new short* [PARTS];
	for (int row = 0; row < PARTS; row++)
		Submatrices[row] = initializeMatrix(COUNT_OF_ROWS/PARTS,COUNT_OF_COLS);
	for (int partsCounter = 0; partsCounter < PARTS; partsCounter++, step += COUNT_OF_ROWS / PARTS) {
		for (int matrixRow=0; matrixRow < COUNT_OF_ROWS / PARTS; matrixRow++, subMatrixRow++) {
			for (int matrixCol = 0; matrixCol < COUNT_OF_COLS; matrixCol++)
				Submatrices[partsCounter][matrixRow * COUNT_OF_COLS + matrixCol] = matrix[subMatrixRow * COUNT_OF_COLS + matrixCol];
		}
	}
	free(matrix);
	return Submatrices;
}

void connectMatrices(short** arrayOfMatrices, short* matrix) {
	int row = 0, int step = 0;
	for (int partsCounter = 0; partsCounter< PARTS; partsCounter++, step += COUNT_OF_ROWS /(2*PARTS)) {
		for (int submatrixRow = 0; row < COUNT_OF_ROWS /(2*PARTS) + step; row++, submatrixRow++) {
			for (int col = 0; col < COUNT_OF_COLS*2; col++)
				matrix[row * COUNT_OF_COLS*2 + col] = arrayOfMatrices[partsCounter][submatrixRow * COUNT_OF_COLS * 2 + col];
		}
	}
}

void transformBigMatrixGPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix) {
	short** arrayOfMatrices = splitInSubmatrices(sourseMatrix);
	short* resultmatrix[PARTS];
	short* sourseMatrixGPU;
	short* resultMatrixGPU;
	float timeCounter;
	float resultTime=0;
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	for (int partsCounter = 0; partsCounter < PARTS; partsCounter++) {
		checkOnError(cudaMalloc((void**)&sourseMatrixGPU, sourseMatrixRow/PARTS * sourseMatrixCol * sizeof(short)));
		checkOnError(cudaMalloc((void**)&resultMatrixGPU, sourseMatrixRow/PARTS * sourseMatrixCol * sizeof(short)));
		checkOnError(cudaMemcpy(sourseMatrixGPU, arrayOfMatrices[partsCounter], sourseMatrixRow/PARTS  * sourseMatrixCol * sizeof(short), cudaMemcpyHostToDevice));
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(sourseMatrixCol / block.x, sourseMatrixRow / block.y);
		if (sourseMatrixRow % block.y != 0) grid.y ++;
		if (sourseMatrixCol % block.x != 0) grid.x ++;
		cudaEventCreate(&startTime);
		cudaEventCreate(&stopTime);
		cudaEventRecord(startTime);
		kernel << <grid, block >> > (sourseMatrixGPU, sourseMatrixRow/PARTS, sourseMatrixCol, resultMatrixGPU);
		cudaEventRecord(stopTime);
		cudaEventSynchronize(stopTime);
		cudaEventElapsedTime(&timeCounter, startTime, stopTime);
		resultTime += timeCounter;
		resultmatrix[partsCounter] = (short*)malloc(sourseMatrixRow / PARTS * sourseMatrixCol * sizeof(short));
		checkOnError(cudaMemcpy(resultmatrix[partsCounter], resultMatrixGPU,
			sourseMatrixRow / PARTS * sourseMatrixCol * sizeof(short),
			cudaMemcpyDeviceToHost));
		checkOnError(cudaFree(sourseMatrixGPU));
		checkOnError(cudaFree(resultMatrixGPU));
	}
	cout << "GPU time: " << resultTime << " ms" << endl;
	connectMatrices(resultmatrix,resultMatrix);
}

__global__ void kernel(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol, short* resultMatrix) {
	int rows =  blockIdx.y * blockDim.y + threadIdx.y;
	int cols = blockIdx.x * blockDim.x  + threadIdx.x;
	if (cols >=sourseMatrixCol || rows >= sourseMatrixRow) return;
	int resultMatrixCol = rows % 2 == 0 ? cols * 2 :cols *2+1;
	resultMatrix[rows/2 * sourseMatrixCol*2 + resultMatrixCol] = sourseMatrix[rows * sourseMatrixCol + cols];
}

void transformMatrixCPU(short* sourseMatrix, int sourseMatrixRow, int sourseMatrixCol,short* resultMatrix){
	auto start_cpu = chrono::steady_clock::now();
	int resultMatrixCol;
	for (int row = 0; row < sourseMatrixRow; row++) {
		for (int col = 0; col < sourseMatrixCol; col++) {
			resultMatrixCol = row % 2 == 0 ? col * 2 : col * 2 + 1;
			resultMatrix[row/2 * sourseMatrixCol*2 + resultMatrixCol] = sourseMatrix[row * sourseMatrixCol + col];
		}
	}
	auto end_cpu = chrono::steady_clock::now();
	cout << "CPU time: " << chrono::duration <double, milli>(end_cpu - start_cpu).count() << " ms" << endl;
}

short* initializeMatrix(int matrixRows, int matrixCols) {
	return (short*)malloc(matrixRows * matrixCols * sizeof(short));
}

void showMatrix(short* matrix,int matrixRows,int matrixCols) {
	for (int i = 0; i < matrixRows; i++) {
		for (int j = 0; j < matrixCols; j++)
			cout << '\t' << matrix[i * matrixCols + j];
		cout << '\n';
	}
}

void randomElements(short* matrix, int matrixRows, int matrixCols) {
	for (int i = 0; i < matrixRows; i++) {
		for (int j = 0; j < matrixCols; j++)
			matrix[i * matrixCols + j] = rand() % 100 + 1;
	}
}

bool compareMatrix(short* matrix1, short* matrix2, int matrixRow, int matrixCol) {
	for (auto i = 0; i < matrixRow; i++)
		for (auto j = 0; j < matrixCol; j++)
			if (matrix1[i * matrixCol + j] != matrix2[i * matrixCol + j])
				return true;
	return false;
}

void checkOnError(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		cout << "CUDA return error code: " << cudaStatus;
		cout << " " << cudaGetErrorString(cudaStatus) << endl;
	}
}