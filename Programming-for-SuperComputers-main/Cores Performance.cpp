#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double getTimeStamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

int main()
    {
        const int N = 100000000;
        double wct_start, wct_end;
        const double s = 2.0;
        const int maxThreads = 16;
        const int numMeasurements = 10;


            // Perform measurements for different thread counts
            for (int numThreads = 1; numThreads <= maxThreads; numThreads ++)
            {
                double totalGflops = 0.0;

                // Run the benchmark multiple times for accurate results
                for (int measurement = 0; measurement < numMeasurements; ++measurement)
                {
                    // Initialize arrays a[] and b[]
                    double a[N], b[N];

                    for (int i = 0; i < N; ++i)
                    {
                        a[i] = 0.; // Initial values of array a[]
                        b[i] = 0.; // Initial values of array b[]
                    }

                    // Measure execution time using chrono library
                    wct_start = getTimeStamp();
                    // Parallelize the update loop with OpenMP
#pragma omp parallel for num_threads(numThreads)
                    for (int i = 0; i < N; ++i)
                        a[i] = s * a[i] + b[i];

                    wct_end = getTimeStamp();
                    double time = (wct_end - wct_start);
                    // Calculate Gflop/s
                    double gflops = (2*N) / (time * 1e6);
                    totalGflops += gflops;
                }

                // Average Gflop/s across measurements
                double avgGflops = totalGflops / numMeasurements;

                // Output the results
                printf("%d\t %f\n", numThreads, avgGflops/1e3);
            }


        return 0;
    }