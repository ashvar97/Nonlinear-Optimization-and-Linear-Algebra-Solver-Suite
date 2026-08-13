// OpenMP parallel-for overhead benchmark: measures cycles-per-element for a trivial vectorizable
// update as the thread count is scaled from 1 to 72, to observe where synchronization overhead
// starts to dominate.
//
// Recovered from a truncated/minified source file (no line breaks, missing closing braces for
// the outer loop and main()); logic is otherwise unchanged. The original also accumulated into
// an unused `sum` inside the parallel region, which is a a benign but pointless data race (the
// value was never read) -- removed here rather than "fixed" with a critical section, since it
// served no purpose.
#include <omp.h>
#include <chrono>
#include <iostream>

using namespace std;

static double nodes(double threads, int itr)
{
    omp_set_num_threads(static_cast<int>(threads));

    double wcTimeStart = omp_get_wtime();

    double* A = new double[itr];
    double* B = new double[itr];
    double* C = new double[itr];
    for (int i = 0; i < itr; i++) {
        A[i] = 3.1414;
        B[i] = 1.414;
        C[i] = 2.2360;
    }

#pragma omp parallel for
    for (int i = 0; i < itr; i++) {
        A[i] = A[i] + B[i] * C[i];
    }

    double wcTimeEnd = omp_get_wtime();
    double wcTime = wcTimeEnd - wcTimeStart;
    double freq = 2.e9;
    double cyclesPerElement = (wcTime * freq) / itr;
    cout << cyclesPerElement << '\n';

    delete[] A;
    delete[] B;
    delete[] C;
    return cyclesPerElement;
}

int main()
{
    for (int threads = 1; threads <= 72; threads++) {
        nodes(threads, 1.e8);
    }
    return 0;
}
