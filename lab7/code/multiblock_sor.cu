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

	// start with some bounds checking to be safe
	if ((threadIdx.x >= 0) && (threadIdx.x < 15))
	{
		if ((threadIdx.y >= 0) && (threadIdx.y < 15))
		{
			
			// variables needed for SOR
			int   i_start, i_end, j_start, j_end;
			float change = 0;

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