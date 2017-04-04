#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define MINVAL 0.00
#define MAXVAL 10.0
#define TOL    1e-5
double CPS =   2.9e9;

////////////////////////////  CUDA RELATED  ////////////////////////////////////


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

__global__ void MMM_kernel(float* A, float* B, float* dst, int len)
{
	const int row = threadIdx.x + blockDim.x * blockIdx.x;
	const int col = threadIdx.y + blockDim.y * blockIdx.y;

	if(((row >= 0) && (row < len)) && ((col >= 0) && (col < len)))
	{
		int k;
		for(k = 0; k < len; k++) dst[row * len + col] = A[row * len + k] * B[k * len + col];
	}
}