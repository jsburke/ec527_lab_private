//////////////////////////////////////////////////////////////////////////////////////
//
//  gcc -O1 -fopenmp -g -o SOR SOR.c -lrt -lm -fno-stack-protector
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GIG 1000000000
#define CPG 2.0           // Cycles per GHz -- Adjust to your computer
double CPS 		  = 2.9e9;
double NO_THREADS = 0;

#define OPTIONS 1
double OMEGA;

typedef double data_t;

#define FILE_PREFIX ((const unsigned char*) "SOR_OMP_")

//  defines for bounding values in array
#define MINVAL 0.00
#define MAXVAL 10.0

#define TOL    0.00001
////////////////  Vector Related /////////////////////////////

/* Create abstract data type for vector -- here a 2D array */
typedef struct {
  long int len;
  data_t *data;
} vec_rec, *vec_ptr;

vec_ptr new_vec(long int len);
int set_vec_length(vec_ptr v, long int index);
long int get_vec_length(vec_ptr v);
int init_vector(vec_ptr v, long int len);
int init_vector_rand(vec_ptr v, long int len);
int print_vector(vec_ptr v);

double fRand(double fMin, double fMax);
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
double ts_sec(struct timespec ts);
struct timespec ts_diff(struct timespec start, struct timespec end);
double measure_cps(void);

///////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	  // handle command line
  int BASE, DELTA, ITERS;

  if(argc != 5)
  {
    printf("must have five arguments\n");
    return 0;
  }

  BASE   	 = strtol(argv[1], NULL, 10);
  DELTA  	 = strtol(argv[2], NULL, 10);
  ITERS  	 = strtol(argv[3], NULL, 10);
  NO_THREADS = strtol(argv[4], NULL, 10);
  
  if(BASE <= 0) {
    printf("BASE must be greater than zero\n");
    return 0;
  }

  if(DELTA <= 0) {
    printf("DELTA must be at least one\n");
    return 0;
  }

  if(ITERS <= 0) {
    printf("ITERS must be at least one\n");
    return 0;
  }

  if(NO_THREADS < 2)
  {
  	printf("NO_THREADS really should be more than two\n");
  	return 0;
  }

  return 0;
}

/////////////////////////////  Timing related  ///////////////////////////////

double ts_sec(struct timespec ts)
{
  return ((double)(ts.tv_sec)) + ((double)(ts.tv_nsec))/1.0e9;
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

  total_time = ts_sec(ts_diff(cal_start, cal_end));
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

/////////////////////////////  vector related /////////////////////////////

vec_ptr new_vec(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  vec_ptr result = (vec_ptr) malloc(sizeof(vec_rec));
  if (!result) return NULL;  /* Couldn't allocate storage */
  result->len = len;

  /* Allocate and declare array */
  if (len > 0) {
    data_t *data = (data_t *) calloc(len*len, sizeof(data_t));
    if (!data) {
    free((void *) result);
    printf("\n COULDN'T ALLOCATE STORAGE \n");
    return NULL;  /* Couldn't allocate storage */
  }
  result->data = data;
  }
  else result->data = NULL;

  return result;
}

/* Set length of vector */
int set_vec_length(vec_ptr v, long int index)
{
  v->len = index;
  return 1;
}

/* Return length of vector */
long int get_vec_length(vec_ptr v)
{
  return v->len;
}

/* initialize 2D vector */
int init_vector(vec_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len*len; i++) v->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize vector with another */
int init_vector_rand(vec_ptr v, long int len)
{
  long int i;

  if (len > 0) {
    v->len = len;
    for (i = 0; i < len*len; i++)
      v->data[i] = (data_t)(fRand((double)(MINVAL),(double)(MAXVAL)));
    return 1;
  }
  else return 0;
}

/* print vector */
int print_vector(vec_ptr v)
{
  long int i, j, len;

  len = v->len;
  printf("\n length = %ld", len);
  for (i = 0; i < len; i++) {
    printf("\n");
    for (j = 0; j < len; j++)
      printf("%.4f ", (data_t)(v->data[i*len+j]));
  }
}

data_t *get_vec_start(vec_ptr v)
{
  return v->data;
}

/************************************/

double fRand(double fMin, double fMax)
{
    double f = (double)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/////////////////////////////////////////////////

double omega_calc(int elements)
{
  double calc = (1.8169 + 0.09 * log(19.6578 * elements) ) / 1.5521;
  return (calc >= 2.00) ? 1.99 : calc;
}

void SOR_basic(vec_ptr v, int *iterations)
{
  long int i, j, row_offset;
  long int length = get_vec_length(v);
  data_t *data = get_vec_start(v);
  double change, mean_change = 100;   // start w/ something big
  int iters = 0;

  while ((mean_change/(double)(length*length)) > (double)TOL) {
    iters++;
    mean_change = 0;
    for (i = 1; i < length-1; i++)  {
      row_offset = i * length;
      for (j = 1; j < length-1; j++) {
        change = data[row_offset+j] - .25 * (data[row_offset-length+j] +
            data[row_offset+length+j] +
            data[row_offset+j+1] +
            data[row_offset+j-1]);
        data[row_offset+j] -= change * OMEGA;
        if (change < 0) {
          change = -change;
        }
        mean_change += change;
      }
    }
    if (abs(data[(length-2)*(length-2)]) > 10.0*(MAXVAL - MINVAL)) {
      printf("\n PROBABLY DIVERGENCE iter = %d", iters);
      break;
    }
  }
  *iterations = iters;
  // printf("\n iters = %d", iters);
}

void SOR_red_black(vec_ptr v, int *iterations)
{
	int      tid, i, j, row_offset, iters = 0;
	long int len   = get_vec_length(v);
	long int len2  = len*len;
	data_t	 *data = get_vec_start(v);
	double	 change, mean_change;

	do{
		mean_change = 0;
		iters ++;
		#pragma omp parallel num_threads((int) NO_THREADS) private(tid, i, j, row_offset, change)
		{
			tid = omp_get_thread_num();
			int row_offset = i * len;

			for(i = 1; i < len - 1; i++)
			{
				for(j = tid; j < len - 1; j += NO_THREADS)  //will create unbalanced loading :/
				{
					change = data[row_offset+j] - .25 * (data[row_offset-len+j] +
			            data[row_offset+len+j] +
			            data[row_offset+j+1] +
			            data[row_offset+j-1]);

			        data[row_offset+j] -= change * OMEGA;
			        change             = (change < 0) ? -change : change;
			        mean_change        += change;
				}
			}
		}
	}while(mean_change/(double)len2 > (double)TOL);
}