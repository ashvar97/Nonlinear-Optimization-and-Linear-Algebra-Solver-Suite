#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>

///////////////////////////////////////////////////////////////////////////////
// Kernel
///////////////////////////////////////////////////////////////////////////////

__global__ void gpuKernel( double *d_C, double *d_A, double *d_B, int DATA_N ) {

  for(int pos=(blockIdx.x*blockDim.x)+threadIdx.x;
          pos < DATA_N;
          pos += blockDim.x*gridDim.x) {
     d_C[pos] = d_A[pos] + d_B[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
double RandFloat(double low, double high){
  double t = (double)rand() / (double)RAND_MAX;
  return (1.0f - t) * low + t * high;
}


#define GPU_ERROR(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    printf( "GPUassert: %s in %s:%d\n", cudaGetErrorString(code), file, line);

    if (abort)
      exit(code);
  }
}

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}
///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
  double *h_A, *h_B, *h_C_CPU, *h_C_GPU;
  double *d_A, *d_B, *d_C;
  double  ws1,we1,ws0,we0;

  int i,blocks=8,threads=8;

  //Total number of data elements
  int    DATA_N =  1000;

  if(argc>1) DATA_N=atoi(argv[1]);
  if(argc>2) blocks=atoi(argv[2]);
  if(argc>3) threads=atoi(argv[3]);

  const int   DATA_SZ = DATA_N * sizeof(double);
  const int RESULT_SZ = DATA_N  * sizeof(double);


  printf("Initializing data...\n");
  printf("...allocating CPU memory.\n");
  GPU_ERROR(cudaHostAlloc( (void**)&h_A, (sizeof (double) * DATA_N), 0)) ;
  GPU_ERROR(cudaHostAlloc( (void**)&h_B, (sizeof (double) * DATA_N), 0)) ;
  GPU_ERROR(cudaHostAlloc( (void**)&h_C_CPU, (sizeof (double) * DATA_N), 0)) ;
  GPU_ERROR(cudaHostAlloc( (void**)&h_C_GPU, (sizeof (double) * DATA_N), 0)) ;

  printf("...allocating GPU memory.\n");
  GPU_ERROR( cudaMalloc((void **)&d_A, DATA_SZ))   ;
  GPU_ERROR(  cudaMalloc((void **)&d_B, DATA_SZ))   ;
  GPU_ERROR(cudaMalloc((void **)&d_C, RESULT_SZ)) ;

  printf("...generating input data in CPU mem.\n");
  srand(123);
  //Generating input data on CPU
  for(i = 0; i < DATA_N; i++){
    h_A[i] = RandFloat(0.0f, 1.0f);
    h_B[i] = RandFloat(0.0f, 1.0f);
  }

  printf("...copying input data to GPU mem.\n");

  ws0 = dtime();

  //Copy data to GPU memory for further processing 
  GPU_ERROR(cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice));
  GPU_ERROR(cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice));
  printf("Data init done. Now executing GPU kernel ...\n");

  GPU_ERROR(cudaDeviceSynchronize()) ;
  ws1 = dtime();
  GPU_ERROR(cudaDeviceSynchronize()) ;

  dim3 threadsPerBlock(threads);           // (x,y) or (x,y,z) works too
  dim3 numBlocks(blocks);                // (x,y) works too
                                        // also ProbSize/threadsPerBlock.x etc.

  //Kernel Call:  <<<BLOCKS, THREADS_PER_BLOCK>>>
  gpuKernel<<<numBlocks,threadsPerBlock>>>(d_C, d_A, d_B, DATA_N);
  GPU_ERROR(cudaPeekAtLastError());
  GPU_ERROR(cudaDeviceSynchronize());
  we1 = dtime();


  printf("Reading back GPU result...\n");
  //Read back GPU results to compare them to CPU results
  GPU_ERROR(cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost)) ;
  we0 = dtime();



  printf("CUDA performance: RAW: %lf MIt/s     incl. transf.: %lf MIt/s\n",
	 (double)DATA_N/(we1-ws1)/1e6,
	 (double)DATA_N/(we0-ws0)/1e6);

  printf("Shutting down...\n");
  cudaFree(d_C) ;
  cudaFree(d_B) ;
  cudaFree(d_A) ;
  cudaFree(h_C_GPU);
  cudaFree(h_C_CPU);
  cudaFree(h_B);
  cudaFree(h_A);
}
