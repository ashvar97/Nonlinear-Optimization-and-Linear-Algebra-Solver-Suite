#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include <math.h>

int main(int argc, char** argv) {

    double wct_start, wct_end;
    int id, nt, k, i, j, M;
    int NITER, N;
    double t = 0;
    const double s = 1.00000000001;

    printf("\n\nVector Stride:  a[i] = b[i] + s * d[i]; \n");
    printf("M\t\t MFlop/s\n\n");


    N = 4e6;
    double* a = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* d = (double*)malloc(N * sizeof(double));

    int M_strides[55] = { 1,2,4,8,10,12,14,17,20,24,29,34,41,50,59,71,86,103,123,148,177,213,256,307,368,442,530,636,763,916,1099,1319,1583,1899,2279,2735,3281,3938,4725,5670,6804,8165,9798,11758,14110,16932,20318,24382,29258,35110,42132,50558,60670,72804,87364 };

    for (k = 0; k < N; ++k) {
        a[k] = 0.;
        b[k] = 0.;
        d[k] = 0.;
    }

    NITER = 1;
    for (j = 0; j < 55; j++) {
        M = M_strides[j];

       
        do {
            // time measurement
            wct_start = getTimeStamp();

            // repeat measurement often enough
            for (k = 0; k < NITER; ++k) {
                // This is the benchmark loop
                for (i = 0; i < N; i += M) {
                    // (b) vector a[i] = b[i] + s * d[i];
                    a[i] = b[i] + s * d[i];
                }
                // end of benchmark loop
                if (a[N / 2] < 0.) printf("%lf", a[N / 2]);
            }
            wct_end = getTimeStamp();
            NITER = NITER * 2;
        } while (wct_end - wct_start < 0.1); // at least 100 ms
     
        NITER = NITER / 2;

        double time = (wct_end - wct_start);
        double MFlop = (2 * N * (NITER / (M)))/(time*1.0e6);
        double MFlopPerSec = MFlop / (1.0e-3 * NITER);

        printf("%d\t %f\n", M, MFlop);


    }
    free(a);
    free(b);
    free(d);

    //}

    return 0;
}


