#include <stdio.h>
#include <stdlib.h>
#include "timing.h"

int main(int argc, char** argv) {

  double wct_start, wct_end;
  int id, nt, k, i;
  int NITER, N;
  double t = 0;
 const double s = 1.00000000001;

  /*if(argc != 2) {
    printf("Usage: %s <size>\n",argv[0]);
    exit(1);
  }*/

  //N = atoi(argv[1]);

/*
    Run your code using the following vector lengths (in elements): N=int(1.5 k), k = 8 ... 44. 
    This will give you a decent resolution and equidistant points on the x axis (see below).
[ 26,38,58,86,130,195,292,438,657,985,1478,2217,3325,4988,7482,11223,16834,25251,37877,
56815,85223,127834,191751,287627,431440,647160,970740,1456110,2184164,3276247,4914370,
7371555,11057332,16585998,24878998,37318497,55977745 ]
*/
  printf("\n\nVector triad: a[i] = b[i] + c[i] * d[i]\n");
  printf("MFlop/s\t\t N\n\n");

  int rangeN[37] = {
    26,38,58,86,130,195,292,438,657,985,1478,2217,3325,4988,7482,11223,16834,25251,37877,
    56815,85223,127834,191751,287627,431440,647160,970740,1456110,2184164,3276247,4914370,
    7371555,11057332,16585998,24878998,37318497,55977745};

 for (int counter = 0; counter < 37; ++counter) {
    
    N=rangeN[counter];
    double *a = (double*)malloc(N * sizeof(double));
    double *b = (double*)malloc(N * sizeof(double));
    double *c = (double*)malloc(N * sizeof(double));
    double *d = (double*)malloc(N * sizeof(double));
    
    for (k = 0; k < N; ++k) {
      a[k] = 0.;
      b[k] = k + 1.;
      c[k] = k + 2.;
      d[k] = k + 3.;
    }

    NITER = 1;
    do {
      // time measurement
      wct_start = getTimeStamp();

      // repeat measurement often enough
      for (k = 0; k < NITER; ++k) {
        // This is the benchmark loop
        for (i = 0; i < N; ++i) {
          // vector triad: a[i] = b[i] + c[i] * d[i]
          a[i] = b[i] + c[i] * d[i];

          // (b) vector update: a[i] = s * a[i]
          //a[i] = s * a[i];
        } 
        // end of benchmark loop
        if (a[N/2] < 0.) printf("%lf", a[N/2]);
      }
      wct_end = getTimeStamp();
      NITER = NITER * 2;
    } while (wct_end - wct_start < 0.1); // at least 100 ms

    NITER = NITER / 2;

    double time = (wct_end - wct_start) / NITER;
    double MFlop = (2.0 * N * NITER) / (time * 1.0e6);
    double MFlopPerSec = MFlop / (1.0e-3 * NITER);

    printf("%f\t %d\n", MFlopPerSec, N);

    free(a);
    free(b);
    free(c);
    free(d);

}

  return 0;
}


