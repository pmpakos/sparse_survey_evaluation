#ifndef SPMM_KERNELS_H
#define SPMM_KERNELS_H

#include "macros/cpp_defines.h"

#include "spmm_bench_common.h"

struct Matrix_Format
{
	char * format_name;
	long m;                         // num rows
	long n;                         // num columns
	long nnz;                       // num non-zeros
	double mem_footprint;
	double csr_mem_footprint;

	virtual void spmm(ValueType * x, ValueType * y, int k) = 0;

	Matrix_Format(long m, long n, long nnz) : m(m), n(n), nnz(nnz)
	{
		csr_mem_footprint = nnz * (sizeof(ValueType) + sizeof(INT_T)) + (m+1) * sizeof(INT_T);
	}
};

struct Matrix_Format * csr_to_format(INT_T * row_ptr, INT_T * col_ind, ValueType * values, long m, long n, long nnz);

#endif /* SPMM_KERNELS_H */
