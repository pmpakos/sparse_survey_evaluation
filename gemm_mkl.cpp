#include <stdio.h>
#include <stdlib.h>

#include "macros/cpp_defines.h"
#include "mkl.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "debug.h"
	#include "time_it.h"
	#include "string_util.h"
	#include "aux/csr_converter.h"
	#include "storage_formats/matrix_market/matrix_market.h"
	#include "storage_formats/dlmc_matrices/dlmc_matrix.h"
#ifdef __cplusplus
}
#endif

#include "bench_common.h"
#include "kernel.h"

int main(int argc, char **argv)
{
    // Following your pattern: <executable> <M> <N> <K>
    if(argc < 4){
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        exit(1);
    }

    int i = 1;
    int M = atoi(argv[i++]);
    int N = atoi(argv[i++]);
    int K = atoi(argv[i++]);

    double time_compute;
    long iterations;

    // Use your standard 64-byte alignment for MKL efficiency 
    ValueType *A = (ValueType *)aligned_alloc(64, M * K * sizeof(ValueType));
    ValueType *B = (ValueType *)aligned_alloc(64, K * N * sizeof(ValueType));
    ValueType *C = (ValueType *)aligned_alloc(64, M * N * sizeof(ValueType));

    // Initialize random data using OpenMP as seen in your lib 
    #pragma omp parallel for
    for(long i=0; i < (long)M * K; ++i) A[i] = (ValueType)rand() / RAND_MAX;
    #pragma omp parallel for
    for(long i=0; i < (long)K * N; ++i) B[i] = (ValueType)rand() / RAND_MAX;
    #pragma omp parallel for
    for(long i=0; i < (long)M * N; ++i) C[i] = 0.0;

    // Define Gemm parameters based on your ValueType [cite: 9]
    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    // Environment variables like your workflow 
    const char* system = getenv("SYSTEM");
    if (system == NULL) system = "Unknown";

    // Warmup iteration 
    #if DOUBLE == 1
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    #else
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
    #endif

    // Benchmark loop following your spmm_bench logic 
    time_compute = 0;
    iterations = 128; 
    for(int i=0; i<iterations; i++){
        time_compute += time_it(1, 
            #if DOUBLE == 1
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
            #else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
            #endif
        );
    }

    // Calculate GFLOPS: 2 * M * N * K operations per GEMM
    double gflops = 2.0 * M * N * K * iterations / time_compute / 1e9;
    
    printf("GEMM kernel - M: %d, N: %d, K: %d, system: %s, gflops: %.2lf\n", 
            M, N, K, system, gflops);

    free(A);
    free(B);
    free(C);

    return 0;
}