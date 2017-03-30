#include <cstdio>
#include <cstdlib>
#include <math.h>

////////////////////////////  CUDA RELATED  ////////////////////////////////////

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

#define ARR_WIDTH			16
#define NUM_ITERS			4
#define NUM_BLOCKS			2
#define THREADS_PER_BLOCK	(ARR_WIDTH/NUM_BLOCKS) + 2 * NUM_ITERS