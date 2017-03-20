/************************************************************************/
// gcc -O1 -fopenmp -g -o test_mmm_inter_omp test_mmm_inter_omp.c -lrt -fno-stack-protector

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GIG 1000000000
#define CPG 2.0           // Cycles per GHz -- Adjust to your computer

#define BASE  0
#define ITERS 10
#define DELTA 96

#define OPTIONS 1
#define IDENT 0

typedef float data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;
/************************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  int clock_gettime(clockid_t clk_id, struct timespec *tp);
  matrix_ptr new_matrix(long int len);
  int set_matrix_length(matrix_ptr m, long int index);
  long int get_matrix_length(matrix_ptr m);
  int init_matrix(matrix_ptr m, long int len);
  int zero_matrix(matrix_ptr m, long int len);
  void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void mmm_jki(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void free_memory(void);

  long int i, j, k;
  long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- MMM \n");

  // declare and initialize the matrix structure
  matrix_ptr a0 = new_matrix(MAXSIZE);
  init_matrix(a0, MAXSIZE);
  matrix_ptr b0 = new_matrix(MAXSIZE);
  init_matrix(b0, MAXSIZE);
  matrix_ptr c0 = new_matrix(MAXSIZE);
  zero_matrix(c0, MAXSIZE);

  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    mmm_ijk(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %lu", i);
  }
   
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    mmm_ijk_omp(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %lu", i);
  }
 
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    mmm_kij(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %lu", i);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    mmm_kij_omp(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    printf("\niter = %lu", i);
  }

  printf("\nlength, ijk, kij, jki");
  for (i = 0; i < ITERS; i++) {
    printf("\n%lu, ", BASE+(i+1)*DELTA);
    for (j = 0; j < OPTIONS; j++) {
      if (j != 0) printf(", ");
      printf("%ld", (long int)((double)(CPG)*(double)
		 (GIG * time_stamp[j][i].tv_sec + time_stamp[j][i].tv_nsec)));
    }
  }

  printf("\n");
  return 0;
}/* end main */

/**********************************************/

/* Create matrix of specified length */
matrix_ptr new_matrix(long int len)
{
  long int i;

  /* Allocate and declare header structure */
  matrix_ptr result = (matrix_ptr) malloc(sizeof(matrix_rec));
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

/* Set length of matrix */
int set_matrix_length(matrix_ptr m, long int index)
{
  m->len = index;
  return 1;
}

/* Return length of matrix */
long int get_matrix_length(matrix_ptr m)
{
  return m->len;
}

/* initialize matrix */
int init_matrix(matrix_ptr m, long int len)
{
  long int i;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(i);
    return 1;
  }
  else return 0;
}

/* initialize matrix */
int zero_matrix(matrix_ptr m, long int len)
{
  long int i,j;

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(IDENT);
    return 1;
  }
  else return 0;
}

data_t *get_matrix_start(matrix_ptr m)
{
  return m->data;
}

/*************************************************/

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

/*************************************************/

/* mmm ijk */
void mmm_ijk(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;

  for (i = 0; i < length; i++)
    for (j = 0; j < length; j++) {
      sum = IDENT;
      for (k = 0; k < length; k++)
	sum += a0[i*length+k] * b0[k*length+j];
      c0[i*length+j] += sum;
    }
}

/* mmm ijk w/ OMP */
void mmm_ijk_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t sum;
  long int part_rows = length/4;

  omp_set_num_threads(4);

#pragma omp parallel shared(a0,b0,c0,length) private(i,j,k,sum)
  {
#pragma omp for
    for (i = 0; i < length; i++) {
      for (j = 0; j < length; j++) {
	sum = IDENT;
	for (k = 0; k < length; k++)
	  sum += a0[i*length+k] * b0[k*length+j];
	c0[i*length+j] += sum;
      }
    }
  }
}

/* mmm kij */
void mmm_kij(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_length(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  for (k = 0; k < length; k++)
    for (i = 0; i < length; i++) {
      r = a0[i*length+k];
      for (j = 0; j < length; j++)
	c0[i*length+j] += r*b0[k*length+j];
    }
}

/* mmm kij w/ omp */
void mmm_kij_omp(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int get_matrix_length(matrix_ptr m);
  data_t *get_matrix_start(matrix_ptr m);
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t r;

  omp_set_num_threads(4);
#pragma omp parallel shared(a0,b0,c0,length) private(i,j,k,r)
  {
#pragma omp for
    for (k = 0; k < length; k++) {
      for (i = 0; i < length; i++) {
	r = a0[i*length+k];
	for (j = 0; j < length; j++)
	  c0[i*length+j] += r*b0[k*length+j];
      }
    }
  }
}

void free_memory(void)
{

}