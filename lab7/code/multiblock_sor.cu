#include <cstdio>
#include <cstdlib>
#include <math.h>

#define MINVAL 0.00
#define MAXVAL 10.0
#define TOL    1e-5

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

__global__ void SOR_kernel(float* arr, int len, float OMEGA)
{

	//  First get the index
	int idx = (threadIdx.x + (blockDim.x * blockIdx.x)) * len + (threadIdx.y + (blockDim.y * blockIdx.y));

	// check if either index places the thread on a fixed element
	if(((idx != 0) && (idx != 2047)) && ((idy != 0) && (idy != 2047)))
	{
		float change = arr[idx] - 0.25 * (arr[idx - len] + arr[idx + len] + arr[idx + 1] + arr[idx - 1]);
		arr[idx] -= change * OMEGA;
	}
}