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

	// if GPU, need to run 1000 iterations more
	for(int i=0; i<1000; i++) MF->spmm(x, y, k);

	time = 0;
	iterations = 128;
	// iterations = 1;
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
