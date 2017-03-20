/*********************************************************************/
// gcc -O1 -fopenmp -o test_omp_for test_omp_for.c -lrt -lm
// export NUM_OMP_THREADS=4
//
// 3 functions, all variations of simple FOR loop program with
// independent iterations.  
// 1. Compute bound -- lots of comptutation on each array element
// 2. Memory bound -- multiple memory references, including pointer
//                 -- following, for each array element
// 3. Overhead bound -- not much work per array element
// Each function also has a serial baseline.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define GIG 1000000000
#define CPG 2.0           // Cycles per GHz -- Adjust to your computer

#define BASE  0
#define ITERS 20
#define DELTA 12

#define OPTIONS 6
#define IDENT 0

#define INIT_LOW -10.0
#define INIT_HIGH 10.0

typedef long int data_t;

/* Create abstract data type for matrix */
typedef struct {
  long int len;
  data_t *data;
} matrix_rec, *matrix_ptr;

/*********************************************************************/
int main(int argc, char *argv[])
{
  int OPTION;
  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp[OPTIONS][ITERS+1];
  matrix_ptr new_matrix(long int len);
  int set_matrix_length(matrix_ptr m, long int index);
  long int get_matrix_length(matrix_ptr m);
  int init_matrix(matrix_ptr m, long int len);
  int zero_matrix(matrix_ptr m, long int len);
  void omp_cb_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c); // comp bound
  void omp_cb(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  void omp_mb_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d); // mem bound
  void omp_mb(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d);
  void omp_ob_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c); // overhead bound
  void omp_ob(matrix_ptr a, matrix_ptr b, matrix_ptr c);
  int init_matrix_rand(matrix_ptr m, long int len);
  int init_matrix_rand_ptr(matrix_ptr m, long int len);

  long int i, j, k;
  long int time_sec, time_ns;
  long int MAXSIZE = BASE+(ITERS+1)*DELTA;

  printf("\n Hello World -- Test OMP \n");

  // declare and initialize the matrix structure
  matrix_ptr a0 = new_matrix(MAXSIZE);
  init_matrix_rand(a0, MAXSIZE);
  matrix_ptr b0 = new_matrix(MAXSIZE);
  init_matrix_rand(b0, MAXSIZE);
  matrix_ptr c0 = new_matrix(MAXSIZE);
  zero_matrix(c0, MAXSIZE);
  matrix_ptr d0 = new_matrix(MAXSIZE);
  init_matrix_rand_ptr(d0, MAXSIZE);

  OPTION = 0;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_cb_bl(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //printf("\niter = %d", i);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_cb(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //    printf("\niter = %d", i);
  }
  
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_mb_bl(a0,b0,c0,d0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //    printf("\niter = %d", i);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_mb(a0,b0,c0,d0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //    printf("\niter = %d", i);
  }
  
  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_ob_bl(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //    printf("\niter = %d", i);
  }

  OPTION++;
  for (i = 0; i < ITERS; i++) {
    set_matrix_length(a0,BASE+(i+1)*DELTA);
    set_matrix_length(b0,BASE+(i+1)*DELTA);
    set_matrix_length(c0,BASE+(i+1)*DELTA);
    clock_gettime(CLOCK_REALTIME, &time1);
    omp_ob(a0,b0,c0);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION][i] = diff(time1,time2);
    //    printf("\niter = %d", i);
  }
  
  printf("\nlength, l^2, ijk, kij, jki");
  for (i = 0; i < ITERS; i++) {
    printf("\n%ld, ", BASE+(i+1)*DELTA);
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

/* initialize matrix to rand */
int init_matrix_rand(matrix_ptr m, long int len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(fRand((double)(INIT_LOW),(double)(INIT_HIGH)));
    return 1;
  }
  else return 0;
}

int init_matrix_rand_ptr(matrix_ptr m, long int len)
{
  long int i;
  double fRand(double fMin, double fMax);

  if (len > 0) {
    m->len = len;
    for (i = 0; i < len*len; i++)
      m->data[i] = (data_t)(fRand((double)(0.0),(double)(i)));
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

double fRand(double fMin, double fMax)
{
  double f = (double)random() / (double)(RAND_MAX);
  return fMin + f * (fMax - fMin);
}

/*************************************************/

/* CPU bound baseline */
void omp_cb_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 0; i < length*length; i++) {
    c0[i] = (data_t)(sqrt(cos(exp((double)(a0[i])))));
  }
}

/* CPU bound openmp */
void omp_cb(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (i = 0; i < length*length; i++) {
    c0[i] = (data_t)(sqrt(cos(exp((double)(a0[i])))));
  }
}

/* memory bound baseline */
void omp_mb_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t *d0 = get_matrix_start(d);

  for (i = 1; i < length*length-1; i++) {
    c0[i] = a0[d0[i]]+b0[d0[i]]+c0[d0[i]]+d0[d0[i]];
  }
}

/* memory bound openmp */
void omp_mb(matrix_ptr a, matrix_ptr b, matrix_ptr c, matrix_ptr d)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);
  data_t *d0 = get_matrix_start(d);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (i = 1; i < length*length-1; i++) {
    c0[i] = a0[d0[i]]+b0[d0[i]]+c0[d0[i]]+d0[d0[i]];
  }
}

/* overhead bound baseline */
void omp_ob_bl(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  for (i = 1; i < length*length-1; i++) {
    c0[i] = a0[i];
  }
}

/* overhead bound openmp */
void omp_ob(matrix_ptr a, matrix_ptr b, matrix_ptr c)
{
  long int i, j, k;
  long int length = get_matrix_length(a);
  data_t *a0 = get_matrix_start(a);
  data_t *b0 = get_matrix_start(b);
  data_t *c0 = get_matrix_start(c);

  omp_set_num_threads(4);
#pragma omp parallel for
  for (i = 1; i < length*length-1; i++) {
    c0[i] = a0[i];
  }
}

