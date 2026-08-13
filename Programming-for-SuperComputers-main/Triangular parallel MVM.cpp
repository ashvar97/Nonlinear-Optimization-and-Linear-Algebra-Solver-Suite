#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

double getTimeStamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

int main() {
    int k, i, NITER;
    double  wct_start, wct_end;
    const int N = 10000;

    printf("Wall time \t Array Size \t Performace (MFlop/s)\n");
    double** a = (double**)malloc(N * sizeof(double*));
    double* b = (double*)malloc(N * sizeof(double));
    double* c = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; ++i) {
        a[i] = (double*)malloc(N * sizeof(double));
    }

#pragma omp parallel for
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c <= r; ++c) {
            a[r][c] = 1.0;  // Example value, you can modify as needed
        }
        b[r] = 1.0;  // Example value, you can modify as needed
        c[r] = 0.0;  // Initialize output vector
    }

        NITER = 1;
        do {
            // time measurement
            wct_start = getTimeStamp();
#pragma omp parallel private(i,k)
            {
                for (k = 0; k < NITER; k++) {
#pragma omp for schedule(runtime)
                    for (int j = 0; j < N; ++j)
                        for (i = 0; i <= j; ++i)
                            c[j] = c[j] + a[i][j] * b[i];
                    if (c[N] < 0.0) printf("%f", c[N]);
                }}
            wct_end = getTimeStamp();
            NITER = NITER * 2;
        } while (wct_end - wct_start < 1.0);
        NITER = NITER / 2;
        double time = (wct_end - wct_start) / 1e3;
        double update_mflops = (2.0 * N * N * NITER) / ((time) * 1e6);
        printf("%f\t %d\t\t %f \n", wct_end - wct_start, N, update_mflops);
        free(a);
        free(b);
        free(c);



        return 0;
}