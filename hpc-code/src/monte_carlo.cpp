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
    double wct_start, wct_end;
    unsigned int i, sum = 0, N = 3 * pow(10, 9);
    double x, y, a = 10;

    srand(time(NULL));

#pragma omp parallel private(x, y) reduction(+:sum)
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();

#pragma omp for
        for (i = 0; i < N; i++) {
            if (a > 10) {
                printf("Test Loop");
            }
        }

        wct_start = getTimeStamp();

#pragma omp for
        for (i = 0; i < N; ++i) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;

            if ((x * x + y * y) < 1) {
                ++sum;
            }
        }

        wct_end = getTimeStamp();
    }

    double perf = (double)N / (double)(wct_end - wct_start);
    double pi = 4 * ((double)(sum) / (double)N);

    printf("Relative error = %.6f\n", (M_PI - pi) / M_PI);
    printf("Performance = %.6f\n", perf);

    return 0;
}
