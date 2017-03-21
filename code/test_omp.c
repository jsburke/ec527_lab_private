/*********************************************************************/
// gcc -O1 -fopenmp -o test_omp test_omp.c -lrt -lm
// export NUM_OMP_THREADS=4
//

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

  long int i, j, k;
  long int time_sec, time_ns;
  char str[] = "Hello World!";

  printf("\n Hello World -- Test OMP \n");

  omp_set_num_threads(4);

#pragma omp parallel for ordered
for (i = 0; i < 12; i++) {
    #pragma omp ordered
    printf("%c", str[i]);
}

  printf("\n");

  return 0;  
}/* end main */

/**********************************************/
