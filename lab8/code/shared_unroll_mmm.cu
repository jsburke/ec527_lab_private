#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#define MINVAL 		0.00
#define MAXVAL 		10.0
#define TOL    		1e-5
#define NUM_THREADS 16
double CPS =   		2.9e9;

int LEN;		// to be defined via cmd args

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


//  	  Unroll factor is 4
__global__ void MMM_kernel(float* A, float* B, float* dst, int len)
{
	__shared__ float Ms [NUM_THREADS][NUM_THREADS];
	__shared__ float Ns [NUM_THREADS][NUM_THREADS];

	int bx, by, tx, ty, row, col;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;

	row = by * NUM_THREADS + ty;
	col = bx * NUM_THREADS + tx;

	float partial = 0;

	for(int k = 0; k < len/NUM_THREADS; k+=4)
	{
		Ms[ty][tx] = A[row *  len + (k * NUM_THREADS + tx)];
		Ms[ty][tx] = A[row *  len + ((k+1) * NUM_THREADS + tx)];
		Ms[ty][tx] = A[row *  len + ((k+2) * NUM_THREADS + tx)];
		Ms[ty][tx] = A[row *  len + ((k+3) * NUM_THREADS + tx)];

		Ns[ty][tx] = B[col + (k * NUM_THREADS + ty) * len];
		Ns[ty][tx] = B[col + ((k+1) * NUM_THREADS + ty) * len];
		Ns[ty][tx] = B[col + ((k+2) * NUM_THREADS + ty) * len];
		Ns[ty][tx] = B[col + ((k+3) * NUM_THREADS + ty) * len];
		__syncthreads();

		for(int r = 0; r < NUM_THREADS; r++)
			partial += Ms[ty][k] * Ns[k][tx];
		__syncthreads();
	}
	
	for(;k < len/NUM_THREADS; k++)  //this loop is the sweeper
	{
		Ms[ty][tx] = A[row *  len + (k * NUM_THREADS + tx)];
		Ns[ty][tx] = B[col + (k * NUM_THREADS + ty) * len];
		__syncthreads();

		for(int r = 0; r < NUM_THREADS; r++)
			partial += Ms[ty][k] * Ns[k][tx];
		__syncthreads();
	}

	dst[row * len + col] = partial;
}

//////////////////////////////  MATRIX  /////////////////////////////////////////

float* matrix_create(int len);
int    matrix_init(float* mat, int len);
int    matrix_zero(float* mat, int len);
int    matrix_copy(float* src, float* dst, int len);
void   MMM_CPU(float* A, float* B, float* dst, int len);

/////////////////  Time related  //////////////////////////////

//rdtsc related
typedef union {
  unsigned long long int64;
  struct {unsigned int lo, hi;} int32;
} mcps_tctr;

#define MCPS_RDTSC(cpu_c) __asm__ __volatile__ ("rdtsc" : \
                     "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))

int clock_gettime(clockid_t clk_id, struct timespec *tp);
struct timespec diff(struct timespec start, struct timespec end);
double ts_ms(struct timespec ts);
struct timespec ts_diff(struct timespec start, struct timespec end);
double measure_cps(void);

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		printf("\nPlease pass a length in.\n");
		return 0;
	}

	LEN = strtol(argv[1], NULL, 10);

	if(LEN <= 0)
	{
		printf("\nLength must be greater than zero\n");
		return 0;
	}

	int size = LEN * LEN * sizeof(float);
	int NUM_BLOCKS = LEN / NUM_THREADS;

	if(LEN % NUM_THREADS != 0)  // die if not a good fit
	{
		printf("\nOdd Numbr of blocks\n");
		return 0;
	}

	// CUDA Timing
	cudaEvent_t start_full, start_mmm, stop_full, stop_mmm;
	float 		d_time_full, d_time_mmm;

	// CPU Timing
	struct timespec time1, time2;
	double h_time;

	// CPU set up
	float *h_A, *h_B, *h_dst_gpu, *h_dst_cpu, *d_A, *d_B, *d_dst;

	measure_cps();

	h_A = matrix_create(LEN);
	if(!h_A) return 0;
	if(!matrix_init(h_A, LEN)) return 0;

	h_B = matrix_create(LEN);
	if(!h_B) return 0;
	if(!matrix_init(h_B, LEN)) return 0;

	h_dst_cpu = matrix_create(LEN);  //  cpu result
	if(!h_dst_cpu) return 0;  
	if(!matrix_zero(h_dst_cpu, LEN)) return 0;  

	h_dst_gpu = matrix_create(LEN);  // gpu result
	if(!h_dst_gpu) return 0;  
	if(!matrix_zero(h_dst_gpu, LEN)) return 0;

	//  GPU Set up

	d_A   = NULL;
	d_B   = NULL;
	d_dst = NULL;

	CUDA_SAFE_CALL(cudaSetDevice(0));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_A, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_B, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst, size));
	
	cudaEventCreate(&start_full);
	cudaEventCreate(&start_mmm);
	cudaEventCreate(&stop_full);
	cudaEventCreate(&stop_mmm);

	// start the GPU calculations

	dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
	dim3 dimGrid(NUM_BLOCKS, NUM_BLOCKS, 1);

	cudaEventRecord(start_full,0);
	CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	cudaEventRecord(start_mmm,0);
	MMM_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_dst, LEN);
	cudaEventRecord(stop_mmm,0);
	cudaEventSynchronize(stop_mmm);

	CUDA_SAFE_CALL(cudaPeekAtLastError());
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(h_dst_gpu, d_dst, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop_full, 0);
	cudaEventSynchronize(stop_full);
	cudaEventElapsedTime(&d_time_mmm, start_mmm, stop_mmm);
	cudaEventElapsedTime(&d_time_full, start_full, stop_full);
	printf("\nGPU MMM  Time: %f ms", d_time_mmm);
	printf("\nGPU FUll Time: %f ms", d_time_full);
	cudaEventDestroy(start_full);
	cudaEventDestroy(stop_full);

	//CPU calculation

	clock_gettime(CLOCK_REALTIME, &time1);
	MMM_CPU(h_A, h_B, h_dst_cpu, LEN);
	clock_gettime(CLOCK_REALTIME, &time2);
	h_time = ts_ms(ts_diff(time1, time2));
	printf("\nCPU Time: %lf ms\n", h_time);

	int i, num_elements;
	num_elements = LEN * LEN;

	for(i = 0; i < num_elements; i++)
	{
		if((h_dst_cpu - h_dst_gpu) > (float) TOL)
		{
			printf("\nResult verification issue at element %d | CPU: %f | GPU: %f\n", i, h_dst_cpu, h_dst_gpu);
			return 0;
		}
	}

	// Free stuff

	CUDA_SAFE_CALL(cudaFree(d_A));
	CUDA_SAFE_CALL(cudaFree(d_B));
	CUDA_SAFE_CALL(cudaFree(d_dst));

	free(h_A);
	free(h_B);
	free(h_dst_gpu);
	free(h_dst_cpu);

	printf("\nDone\n");

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
	float* arr;
	if(len > 0)
	{
		arr = (float*) calloc(len*len, sizeof(float));
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
	printf("\nError in initializing matrix\n");
	return 0;
}

int matrix_zero(float* mat, int len)
{
	int len_sq, i;

	if(len > 0)
	{
		len_sq = len * len;
		for(i = 0; i < len_sq; i++)
		{
			mat[i] = 0;
		}
		return 1;
	}
	printf("\nFailed to zero matrix\n");
	return 0;
}

int   matrix_copy(float* src, float* dst, int len)
{
	int len_sq, i;

	if(len > 0)
	{
		len_sq = len * len;
		for(i = 0; i < len_sq; i++)
		{
			dst[i] = src[i];
		}
		return 1;
	}
	printf("\nFailed to copy matrix\n");
	return 0;
}

void	MMM_CPU(float* A, float* B, float* dst, int len)
{
	
}

/////////////////////////////  Timing related  ///////////////////////////////

double ts_ms(struct timespec ts)
{
  return ((((double)(ts.tv_sec))*1.0e9) + ((double)(ts.tv_nsec)))/(1.0e6);
}

/* ---------------------------------------------------------------------------
| Make the CPU busy, and measure CPS (cycles per second).
|
| Explanation:
| If tests are very fast, they can run so quickly that the SpeedStep control
| (in kernel and/or on-chip) doesn't notice in time, and the first few tests
| might finish while the CPU is still in its sleep state (about 800 MHz,
| judging from my measurements)
|   A simple way to get around this is to run some kind of busy-loop that
| forces the OS and/or CPU to notice it needs to go to full clock speed.
| We print out the results of the computation so the loop won't get optimised
| away.
|
| Copy this code into other programs as desired. It provides three entry
| points:
|
| double ts_sec(ts): converts a timespec into seconds
| timespec ts_diff(ts1, ts2): computes interval between two timespecs
| measure_cps(): Does the busy loop and prints out measured CPS (cycles/sec)
--------------------------------------------------------------------------- */

struct timespec ts_diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

double measure_cps()
{
  struct timespec cal_start, cal_end;
  mcps_tctr tsc_start, tsc_end;
  double total_time;
  double total_cycles;
  /* We perform a chaotic iteration and print the result, to defeat
     compiler optimisation */
  double chaosC = -1.8464323952913974; double z = 0.0;
  long int i, ilim, j;

  /* Do it twice and throw away results from the first time; this ensures the
   * OS and CPU will notice it's busy and set the clock speed. */
  for(j=0; j<2; j++) {
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cal_start);
    MCPS_RDTSC(tsc_start);
    ilim = 50*1000*1000;
    for (i=0; i<ilim; i++)
      z = z * z + chaosC;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cal_end);
    MCPS_RDTSC(tsc_end);
  }

  total_time = ts_ms(ts_diff(cal_start, cal_end));
  total_cycles = (double)(tsc_end.int64-tsc_start.int64);
  CPS = total_cycles / total_time;
  printf("z == %f, CPS == %g\n", z, CPS);

  return CPS;
}
/* ---------------------------------------------------------------------------
| End of measure_cps code
--------------------------------------------------------------------------- */

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
