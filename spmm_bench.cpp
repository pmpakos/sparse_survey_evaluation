#include <stdio.h>
#include <stdlib.h>

#include "macros/cpp_defines.h"

#ifdef __cplusplus
extern "C"{
#endif
	#include "debug.h"
	#include "time_it.h"
	#include "string_util.h"
	#include "aux/csr_converter.h"
	#include "storage_formats/matrix_market/matrix_market.h"
#ifdef __cplusplus
}
#endif

#include "bench_common.h"
#include "kernel.h"

// Utils macro
#define Min(x,y) ((x)<(y)?(x):(y))
#define Max(x,y) ((x)>(y)?(x):(y))
#define Abs(x) ((x)>(0)?(x):-(x))

void CheckAccuracy(INT_T * row_ptr, INT_T * col_idx, ValueType * val, INT_T m, INT_T n, INT_T k, ValueType * x, ValueType * y)
{
	__attribute__((unused)) ValueType epsilon_relaxed = 1e-4;
	#if DOUBLE == 0
		ValueType epsilon = 1e-5;
	#elif DOUBLE == 1
		ValueType epsilon = 1e-10;
	#endif
	long i;
	ValueType * y_gold = (typeof(y_gold)) malloc(m * k * sizeof(*y_gold));
	ValueType * y_test = (typeof(y_test)) malloc(m * k * sizeof(*y_test));
	#pragma omp parallel
	{
		ValueType sum;
		long i, j;
		#pragma omp for
		for(i=0;i<m * k;i++)
		{
			y_gold[i] = 0;
			y_test[i] = y[i];
		}
		#pragma omp for
		for (i = 0; i < m; i++) {
			for (long c = 0; c < k; c++) {
				ValueType value, tmp, compensation;
				compensation = 0;
				sum = 0;
				for (j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
					value = val[j] * x[c * n + col_idx[j]] - compensation;
					tmp = sum + value;
					compensation = (tmp - sum) - value;
					sum = tmp;
				}
				y_gold[i * k + c] = sum;
			}
		}
	}

	ValueType maxDiff = 0, diff;
	for(i=0;i<m * k;i++)
	{
		diff = Abs(y_gold[i] - y_test[i]);
		if (y_gold[i] > epsilon)
		{
			diff = diff / abs(y_gold[i]);
			maxDiff = Max(maxDiff, diff);
		}
	}
	if(maxDiff > epsilon)
		printf("Test failed! (%g)\n", (double)maxDiff);

	free(y_gold);
	free(y_test);
}

int main(int argc, char **argv)
{
	if(argc<3){
		printf("Usage: %s <matrix_market_file> <k>\n", argv[0]);
		exit(1);
	}

	int i = 1;
	double time;
	
	struct Matrix_Market * MTX = NULL;
	ValueType * coo_val = NULL;   // MATRIX_MARKET_FLOAT_T is always double, as reference for error calculation.
	INT_T * coo_rowind = NULL;
	INT_T * coo_colind = NULL;
	long coo_m = 0;
	long coo_n = 0;
	long coo_nnz = 0;

	ValueType * csr_a = NULL;   // values (of size NNZ)
	INT_T * csr_ia = NULL;
	INT_T * csr_ja = NULL;
	long csr_m = 0;
	long csr_n = 0;
	long csr_nnz = 0;
	
	struct Matrix_Format * MF;   // Real matrices.

	ValueType * x, * y;
	long iterations;

	char * file_in;
	file_in = argv[i++];
	char matrix_name[1000];
	snprintf(matrix_name, sizeof(matrix_name), "%s", file_in);

	int k = atoi(argv[i++]);

	time = time_it(1,
		long expand_symmetry = 1;
		long pattern_dummy_vals = 1;
		MTX = mtx_read(file_in, expand_symmetry, pattern_dummy_vals);
		coo_rowind = MTX->R;
		coo_colind = MTX->C;
		coo_m = MTX->m;
		coo_n = MTX->n;
		coo_nnz = MTX->nnz;
		mtx_values_convert_to_real(MTX);
		coo_val = (typeof(coo_val)) MTX->V;
		MTX->R = NULL;
		MTX->C = NULL;
		MTX->V = NULL;
		mtx_destroy(&MTX);
	);
	printf("time read: %lf\n", time);

	time = time_it(1,
		csr_a = (typeof(csr_a)) aligned_alloc(64, coo_nnz * sizeof(*csr_a));
		csr_ja = (typeof(csr_ja)) aligned_alloc(64, coo_nnz * sizeof(*csr_ja));
		csr_ia = (typeof(csr_ia)) aligned_alloc(64, (coo_m+1) * sizeof(*csr_ia));
		csr_m = coo_m;
		csr_n = coo_n;
		csr_nnz = coo_nnz;
		_Pragma("omp parallel for")
		for (long i=0;i<coo_nnz;i++)
		{
			csr_a[i] = 0.0;
			csr_ja[i] = 0;
		}
		_Pragma("omp parallel for")
		for (long i=0;i<coo_m+1;i++)
			csr_ia[i] = 0;
		coo_to_csr(coo_rowind, coo_colind, coo_val, coo_m, coo_n, coo_nnz, csr_ia, csr_ja, csr_a, 1, 0);

		free(coo_rowind);
		free(coo_colind);
		free(coo_val);
	);
	printf("time coo to csr: %lf\n", time);

	time = time_it(1,
		MF = csr_to_format(csr_ia, csr_ja, csr_a, csr_m, csr_n, csr_nnz);
	);
	printf("time convert to format: %lf\n", time);

	x = (typeof(x)) aligned_alloc(64, csr_n * k * sizeof(*x));
	#pragma omp parallel for
	for(long i=0; i<csr_n * k; ++i)
		x[i] = 1.0;
	y = (typeof(y)) aligned_alloc(64, csr_m * k * sizeof(sizeof(*y)));
	#pragma omp parallel for
	for(long i=0; i<csr_m * k; i++)
		y[i] = 0.0;

	// warmup iteration
	MF->spmm(x, y, k);

	printf("---\n");
	for(int i=0; i<10; i++){
		printf("i=%d\t[ ", i);
		for(int j=0; j<10; j++) 
			printf("%lf ", y[i*k+j]);
		printf("]\n");
	}
	printf("---\n");
	CheckAccuracy(csr_ia, csr_ja, csr_a, csr_m, csr_n, k, x, y);

	// if GPU, need to run 1000 iterations more for warmup
	int gpu_kernel = 0;
	const char* env_gpu_kernel = getenv("GPU_KERNEL");
	if (env_gpu_kernel != NULL) {
		gpu_kernel = atoi(env_gpu_kernel);
	} else {
		// handle the case where the environment variable is not set
		fprintf(stderr, "Environment variable GPU_KERNEL not set.\n");
		exit(EXIT_FAILURE);
	}
	if(gpu_kernel)
		for(int i=0; i<1000; i++)
			MF->spmm(x, y, k);

	time = 0;
	iterations = 128;
	for(int i=0; i<iterations; i++){
		time += time_it(1, 
			MF->spmm(x, y, k);
		);
	}
	double gflops = 2.0 * MF->nnz * k * iterations / time / 1e9;
	printf("SpMM kernel - matrix: %s, format: %s, k: %d, gflops: %.2lf\n", matrix_name, MF->format_name, k, gflops);

	free(x);
	free(y);
	free(csr_a);
	free(csr_ia);
	free(csr_ja);

	return 0;
}
