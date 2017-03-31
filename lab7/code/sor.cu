#include <cstdio>
#include <cstdlib>
#include <math.h>

#define MINVAL 0.00
#define MAXVAL 10.0

////////////////////////////  CUDA RELATED  ////////////////////////////////////

#define 

#define IMUL(a, b) __mul24(a, b)

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

/////////////////////////////  MATRIX STUFF  ////////////////////////////////////////

float* matrix_create(int len);
void   matrix_init(float* mat, int len);

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int LENGTH = 2048;

	// Create and initialize a 2D array
	float *arr, *d_arr, *d_res, *h_res;
	int    size = LENGTH * LENGTH;

	// initialize array on host
	arr = matrix_create(LENGTH);
	if(!arr) return 0;

	if(!matrix_init(arr, LENGTH))
	{
		printf("\n\tFailed to initialize matrix\n");
		return 0;
	}

	// send array to GPU
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_arr, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice));

	// create single block of 16x16 threads
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(1, 1, 1);

	// invoke kernel, harcode OMEGA
	SOR_kernel<<<dimGrid, dimBlock>>>(d_arr, LENGTH, 1.97);

	// transfer back to CPU
	CUDA_SAFE_CALL(cudaMemcpy(h_arr, d_arr, size, cudMemcpyDeviceToHost));

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////  MATRIX IMPLEMENTATIONS  ////////////////////////////////////////
float float_rand(float min, float max)
{
	float f = (float)random()/RAND_MAX;
	return  min + f * (max - min);
}


float* matrix_create(int len)
{
	if(len > 0)
	{
		float* arr = (float*) calloc(len*len, sizeof(float));
		if(!arr)
		{
			printf("\n\tFailed to allocate array\n");
			return NULL;
		}
	}
	else return NULL;

	return arr;
}

int matrix_init(float* mat, int len)
{
	int len_sq, i;

	if(len > 0)
	{
		len_sq = len * len;
		for (i = 0; i < len_sq; i++)
		{
			mat[i] = float_rand(MINVAL, MAXVAL);
		}
		return 1;
	}
	return 0;
}