#ifndef HC_SPMM_V2_H
#define HC_SPMM_V2_H

#define BLK_H 16 
#define BLK_W 8
#define WARP_SIZE 32

// #define WPBMore 6
#define WPB 3
#define MAX_BLK 3
// #define BLOCKNUM 196133
// #define TMPSIZE 5
#define S_SIZE 62

#include <stdint.h>

/************************************************************************/
/* FUNCTION DECLARATIONS */

void preprocess_gpu_wrapper(int *row_ptr, int *col_idx,  int m, int n, int nnz,
	int * num_row_windows_out, int * blockSize_h_out, int * blockSize_w_out,
	int **nodePointer_ptr_out, int **edgeList_ptr_out, int **blockPartition_ptr_out, int **edgeToColumn_ptr_out, int **edgeToRow_ptr_out, int **hybrid_type_ptr_out, int **row_nzr_ptr_out, int **col_nzr_ptr_out,
	int *nodePointer_size_out, int *edgeList_size_out, int *blockPartition_size_out, int *edgeToColumn_size_out, int *edgeToRow_size_out, int *hybrid_type_size_out, int *row_nzr_size_out, int *col_nzr_size_out);

/************************************************************************/

// spmm_forward_fixed32 --> spmm_forward_plus_fixed32 --> spmm_forward_cuda_kernel_arbi_warps_hybrid_32
__global__ void spmm_forward_cuda_kernel_arbi_warps_hybrid_32(
		const int * __restrict__ nodePointer,		// node pointer.
		const int *__restrict__ edgeList,			// edge list.
		const int *__restrict__ blockPartition, 	// number of TC_blocks (16x8) in each row_window.
		const int *__restrict__ edgeToColumn, 		// eid -> col within each row_window.
		const int *__restrict__ edgeToRow, 		    // eid -> col within each row_window.
		const float *__restrict__ valuesA,
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		const float *__restrict__ input,		    // input feature matrix.
		float *output,							    // aggreAGNNed output feature matrix.
		const int *__restrict__ hybrid_type,
		const int *__restrict__ row_nzr,
		const int *__restrict__ col_nzr
		);

/************************************************************************/

#endif // HC_SPMM_V2_H
