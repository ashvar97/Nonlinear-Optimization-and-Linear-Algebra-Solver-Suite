#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double getTimeStamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

int main()
{
    double wct_start, wct_end;
    int id, nt, k, i;
    int NITER, N;
    double t = 0;
    const double s = 1.00000000001;
    const int maxThreads = 16;
    const int numMeasurements = 10;
    int totalGflops;

    // Loop length N range: 10 to 100,000,000
    for (int N = 10; N <= 100000000; N *= 1.2)
    {
        std::cout << "Loop length (N): " << N << std::endl;

        // Perform measurements for different thread counts
        for (int numThreads = 1; numThreads <= maxThreads; numThreads *= 2)
        {
            double totalGflops = 0.0;
            NITER = 1;
            // Run the benchmark multiple times for accurate results

                // Initialize arrays a[] and b[]
                double a[N], b[N];

                for (int i = 0; i < N; ++i)
                {
                    a[i] = 0.; // Initial values of array a[]
                    b[i] = 0.; // Initial values of array b[]
                }
                // Measure execution time using chrono library
                NITER = 1;
                do {
                    // Time measurement
                    wct_start = getTimeStamp();

                    // Repeat measurement often enough
                    for (k = 0; k < NITER; ++k) {
                        // This is the benchmark loop
#pragma omp parallel for num_threads(numThreads)
                        for (i = 0; i < N; ++i) {
                            // Update benchmark
                            a[i] = s * a[i] + b[i];
                        }
                        // End of benchmark loop
                        if (a[N / 2] < 0.)
                            printf("%lf", a[N / 2]);
                    }

                    wct_end = getTimeStamp();
                    NITER = NITER * 2;
                } while (wct_end - wct_start < 0.2);  // At least 100 ms

                NITER = NITER / 2;
                double time = (wct_end - wct_start) / NITER;
                double MFlop = (2.0 * N * NITER) / (time * 1.0e9);
                double MFlopPerSec = MFlop / (1.0e-3 * NITER);
            // Output the results
            std::cout << "Threads: " << numThreads << ", Gflop/s: " << MFlopPerSec << std::endl;
        }
    }
    return 0;
}
