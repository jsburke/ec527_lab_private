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

__global__ void SOR(float* arr, int len, float OMEGA)
{

	// start with some bounds checking to be safe
	if ((threadIdx.x >= 0) && (threadIdx.x < 15))
	{
		if ((threadIdx.y >= 0) && (threadIdx.y < 15))
		{
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
	int LENGTH;

	if(argc != 2)
	{
		printf("\n\tProgram requires input length\n");
		return 0;
	}

	LENGTH = strtol(argv[1], NULL, 10);

	if(LENGTH <= 0)
	{
		printf("\n\tLENGTH must be greater than zero\n");
		return 0;
	}

	// Create and initialize a 2D array

	float* arr = matrix_create(LENGTH);
	if(!arr) return 0;

	if(!matrix_init(arr, LENGTH))
	{
		printf("\n\tFailed to initialize matrix\n");
		return 0;
	}



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