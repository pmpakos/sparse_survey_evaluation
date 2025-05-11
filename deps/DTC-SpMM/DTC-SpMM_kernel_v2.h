#ifndef DTC_SPMM_V2_H
#define DTC_SPMM_V2_H

#define BLK_H 16 
#define BLK_W 8
#define WARP_SIZE 32

// #define TCBLOCK_PER_WARP 256 
// #define TCBLOCK_PER_WARP 128 
#define TCBLOCK_PER_WARP 64 
// #define TCBLOCK_PER_WARP 32

#define FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#include <stdint.h>

/************************************************************************/
/* FUNCTION DECLARATIONS */

void preprocess_gpu_wrapper(int *row_ptr, int *col_idx,  int m, int n, int nnz, 
	int * num_row_windows_out, int * blockSize_h_out, int * blockSize_w_out, 
	int **RowWindowOffset_ptr_out, int **TCblockRowid_ptr_out, uint8_t **TCblocktileId_ptr_out, int **TCblockoffset_ptr_out, int **SparseAToXindex_ptr_out,
	int *RowWindowOffset_size_out, int *TCblockRowid_size_out, int *TCblocktileId_size_out, int *TCblockoffset_size_out, int *SparseAToXindex_size_out);

/************************************************************************/

// spmm_forward_ptx_uint8_improved --> spmm_forward_improved_ptx_uint8_cuda --> 
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
		const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
		const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		const float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2(
		const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
		const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split(
		const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
		const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4(
		const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
		const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
		const int *__restrict__ Rowwindow_offset, 		// rowid of each TC block.
		const uint8_t *__restrict__ TCblocktile_id, 		// rowid of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);


// spmm_balance_forward_ptx_uint8_prefetch --> spmm_balance_forward_cuda_ptx_unit8_prefetch
__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv(
		const int *__restrict__ TCblock_rowid, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		// colid of each TC block nonzero element.
		const int tc_count,
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		const float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance(
		const int *__restrict__ TCblock_rowid, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int tc_count,
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);


// spmm_forward_ptx_uint8_improved_for_gcn --> spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float4_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.	
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		float *input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const int numNodes,
		const int numEdges,
		const int embedding_dim,				    // embedding dimension.
		const float *__restrict__ input,		    // input feature matrix.
		float *output							    // output feature matrix.
		);


/************************************************************************/

#endif // DTC_SPMM_V2_H
