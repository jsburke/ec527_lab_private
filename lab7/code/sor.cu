#include <cstdio>
#include <cstdlib>
#include <math.h>

#define LEN  2048
#define SIZE LEN*LEN

void initializeArray2D(float *arr, int len, int seed);

/////////////////////////////  CUDA  ////////////////////////////////////////////////////////

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define PRINT_TIME 				1
#define TOL						1e-5

__global__ void SOR_kernel(int len, float* arr) 
{

	// start with some bounds checking to be safe
	if ((threadIdx.x >= 0) && (threadIdx.x < 15))
	{
		if ((threadIdx.y >= 0) && (threadIdx.y < 15))
		{
			
			// variables needed for SOR
			int   i_start, i_end, j_start, j_end;
			float change = 0;
			float OMEGA  = 1.97;

			// set start point for threads
			if (threadIdx.x == 0) i_start = 1;
			else				  i_start = threadIdx.x * 128;

			if (threadIdx.y == 0) j_start = 1;
			else				  j_start = threadIdx.y * 128;

			// set end point for threads
			if (threadIdx.x == 15) i_end = 2046;
			else                   i_end = threadIdx.x * 128 + 127;

			if (threadIdx.y == 15) j_end = 2046;
			else                   j_end = threadIdx.y * 128 + 127;

			//  begin the SOR this portion is responsible for

			int i,j,k;

			for (k = 0; k < 2000; k++)  //2k iterations of SOR
			{
				for (i = i_start; i <= i_end; i++)
				{
					for (j = j_start; j <= j_end; j++)
					{
						change = arr[i*len+j] - 0.25 * (arr[(i-1)*len+j] + arr[(i+1)*len+j] + arr[i*len+j+1] + arr[i*len+j-1]);

						__syncthreads();
						arr[i*len+j] -= change * OMEGA;
						__syncthreads();
					}
				}
			}
		}
	}
}

int main(int argc, char **argv){
		
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	// Arrays on GPU global memoryc
	float *d_mat;

	// Arrays on the host memory
	float *h_mat, h_gpu_res;
	
	int i, errCount = 0, zeroCount = 0;
	
	
    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

	// Allocate GPU memory
	size_t allocSize = SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mat, allocSize));
		
	// Allocate arrays on host memory
	h_mat                        = (float *) malloc(allocSize);
	h_gpu_res					 = (float *) malloc(allocSize);
	
	// Initialize the host arrays
	initializeArray2D(h_mat, LEN, 2453);	
	
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_mat, h_mat, allocSize, cudaMemcpyHostToDevice));
	  
	// Launch the kernel

	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(1, 1, 1);
	SOR_kernel<<<dimGrid, dimBlock>>>(len, d_mat);

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_gpu_res, d_mat, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	
	
	// Compare the results
	// for(i = 0; i < arrLen; i++) {
	// 	if (abs(h_result_gold[i] - h_result[i]) > TOL) {
	// 		errCount++;
	// 	}
	// 	if (h_result[i] == 0) {
	// 		zeroCount++;
	// 	}
	// }
	
	/*
	for(i = 0; i < 50; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_result_gold[i], h_result[i]);
	}
	*/
	
	// if (errCount > 0) {
	// 	printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	// }
	// else if (zeroCount > 0){
	// 	printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	// }
	// else {
	// 	printf("\nTEST PASSED: All results matched\n");
	// }
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_mat));
		   
	free(h_mat);
	free(h_gpu_res);
		
	return 0;
}

void initializeArray2D(float *arr, int len, int seed) {
	int i, j;
	srand(seed);

	for (i = 0; i < len; i++){
		for(j = 0; j < len; j++)
			arr[i] = (float) rand();
	}
}
