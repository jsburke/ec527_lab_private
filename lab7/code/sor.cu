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

__global__ void SOR_kernel(float* arr, float* res, int len, float OMEGA)
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

						//__syncthreads();

						arr[i*len+j] -= change * OMEGA;

						//__syncthreads();
					}
				}
			}

			// copy to result
			for(i = i_start; i <= i_end; i++)
			{
				for(j = j_start; j <=j_end; j++)
				{
					res[i * len + j] = arr[i * len +j];
				}
			}

		}
	}
}

/////////////////////////////  MATRIX STUFF  ////////////////////////////////////////

float* matrix_create(int len);
int    matrix_init(float* mat, int len);
int    matrix_zero(float* mat, int len);
void   SOR_CPU(float* mat, int len, float OMEGA);

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

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int   LEN   = 2048;
	int   size  = LEN * LEN * sizeof(float);
	float OMEGA = 1.97;

	// CUDA Timing
	cudaEvent_t start, stop;
	float d_time;

	//CPU timing
	struct timespec time1, time2;
	double h_time;

	float *h_mat, *d_mat, *h_res, *d_res;

	// set up matrix on host

	measure_cps();

	h_mat = matrix_create(LEN);
	if(!h_mat) return 0;
	if(!matrix_init(h_mat, LEN)) return 0;

	h_res = matrix_create(LEN);
	if(!h_res) return 0;
	if(!matrix_zero(h_res, LEN)) return 0;

	// set up device

	d_mat = NULL;

    CUDA_SAFE_CALL(cudaSetDevice(0));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_mat, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_res, size));
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	CUDA_SAFE_CALL(cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice));

	// Launch the kernel

	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(1, 1, 1);

	SOR_kernel<<<dimGrid, dimBlock>>>(d_mat, d_res, LEN, OMEGA);

	CUDA_SAFE_CALL(cudaPeekAtLastError());
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	// Transfer the results back to the host

	CUDA_SAFE_CALL(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_time, start, stop);
	printf("\nGPU time: %f (msec)\n", d_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// CPU SOR and comparison

	clock_gettime(CLOCK_REALTIME, &time1);
	SOR_CPU(h_mat, LEN, OMEGA);
	clock_gettime(CLOCK_REALTIME, &time2);
	h_time = ts_ms(ts_diff(time1, time2));
	printf("\nCPU timeL %lf (msec)\n", h_time);

	int i, num_elements;
	num_elements = LEN * LEN;

	for(i = 0; i < num_elements; i++)
	{
		if((h_mat - h_res) > (float) TOL)
		{
			printf("\nResult verification failed at element %d\n", i);
			return 0;
		}
	}
	
	// Free stuff

	CUDA_SAFE_CALL(cudaFree(d_mat));

	free(h_res);
	free(h_mat);

	printf("\nDone\n");
	return 0;

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

int   matrix_zero(float* mat, int len)
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

void SOR_CPU(float* mat, int len, float OMEGA)
{
	int   i, j, k;
	float change = 0;
	int   q_idx;

	for(k = 0; k < 2000; k++)
	{
		for(i = 0; i < len; i++)
		{
			for(j = 0; j < len; j++)
			{
				q_idx  = i * len + j;
				change = mat[q_idx] - 0.25 * (mat[q_idx-len] + mat[q_idx+len] + mat[q_idx-1] +mat[q_idx+1]);

				mat[q_idx] -= change * OMEGA;
			}
		}
	}
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
