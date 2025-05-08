#ifndef ACC_SPMM_V2_H
#define ACC_SPMM_V2_H

#include "common.h"
#include "utils.h"
#include "class.h"

/************************************************************************/
/* FUNCTION DECLARATIONS */

__device__ __forceinline__
void tf32_m16n8k8(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC);

__device__ __forceinline__
void tf32_m16n8k8_detail(MAT_VAL_TYPE MatA0, MAT_VAL_TYPE MatA1, MAT_VAL_TYPE MatA2, MAT_VAL_TYPE MatA3, MAT_VAL_TYPE MatB0,MAT_VAL_TYPE MatB1, MAT_VAL_TYPE C0,MAT_VAL_TYPE C1,MAT_VAL_TYPE C2,MAT_VAL_TYPE C3);

__device__ __forceinline__
void tf32_m16n8k4(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC);

__device__ __forceinline__
void wait_group();

/*===================WARP TOOLS====================*/
// static __device__ __forceinline__
// vint lane_id() {
// 	unsigned r;
// 	asm volatile("mov.u32 %0, %laneid;" : "=r"(r));
// 	return r;
// }

// static __device__ __forceinline__ 
// int lane_mask_lt() {
// 	int mask;
// 	asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
// 	return mask;
// }

// static __device__ __forceinline__ 
// __device__ __forceinline__ 
// vint warp_id();

__device__ __forceinline__
void async_copy (MAT_PTR_TYPE shared_addr, const MAT_VAL_TYPE* val);

__device__ __forceinline__
void async_copy_idx (MAT_PTR_TYPE shared_addr, const vint* val);

// Cache in L1, L2.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global(const MAT_VAL_TYPE* a);

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global_cs(const MAT_VAL_TYPE* a);

// Don't cache and fetch again.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global2shared(const MAT_VAL_TYPE* a);

__device__ __forceinline__ 
vint load_int_from_global(const vint* a);

// recently least used cache friendly
__device__ __forceinline__
void store_fp32_to_global(MAT_VAL_TYPE* a, MAT_VAL_TYPE v);

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_shared1(const MAT_PTR_TYPE a);

__device__ __forceinline__
float4 vector_fetch_fp32V4(const float4 *ptr);

__device__ __forceinline__
float2 vector_fetch_fp32V2(const float2 *ptr);

__device__ __forceinline__
MAT_VAL_TYPE load_int_from_shared(const MAT_PTR_TYPE a);

__device__ __forceinline__
float2 ld_shared_float2(uint a);

__device__ __forceinline__
float4 ld_shared_float4(uint a);

__device__ __forceinline__ 
uint getSMId();


__global__
void tf32_computeX128Transpose( 
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_PTR_TYPE* __restrict__    d_block2Idx,
	const MAT_PTR_TYPE* __restrict__    d_data2Idx,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE* d_MatC,
	const vint numNodes,
	const vint feature_dim
);
__global__
void tf32_computeX128TransposePipe2(
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_PTR_TYPE* __restrict__    d_block2Idx,
	const MAT_PTR_TYPE* __restrict__    d_data2Idx,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE* d_MatC,
	const vint numNodes,
	const vint feature_dim
);
__global__
void tf32_computeX128TransposePipe2AdpLoadBalance(
	const MAT_PTR_TYPE* __restrict__    d_group_offset,
	const MAT_PTR_TYPE* __restrict__    d_tc_offset,
	const MAT_PTR_TYPE* __restrict__    d_row_indices,
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE*                       d_MatC,
	const vint numNodes,
	const vint feature_dim
);
__global__
void tf32_computeX128TransposePipe2G128( 
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_PTR_TYPE* __restrict__    d_block2Idx,
	const MAT_PTR_TYPE* __restrict__    d_data2Idx,
	MAT_VAL_TYPE* d_MatB, 
	MAT_VAL_TYPE* d_MatC,
	const vint numNodes,
	const vint feature_dim
);
__global__
void tf32_computeX128TransposePipe2AdpBalanceG128(  
	const MAT_PTR_TYPE* __restrict__    d_group_offset,
	const MAT_PTR_TYPE* __restrict__    d_tc_offset,
	const MAT_PTR_TYPE* __restrict__    d_row_indices,
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE*                       d_MatC,
	const vint numNodes,
	const vint feature_dim
); 
__global__
void tf32_computeX128TransposeAdpBalancePipe(      // 4 个 warp 计算 128 维；8 个 warp / block 计算 256；16 个 warp / block 计算 512 维
	const MAT_PTR_TYPE* __restrict__    d_group_offset,
	const MAT_PTR_TYPE* __restrict__    d_tc_offset,
	const MAT_PTR_TYPE* __restrict__    d_row_indices,
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE*                       d_MatC,
	const vint numNodes,
	const vint feature_dim
);
__global__
void tf32_computeX128TransposeAdpBalancePipeG128(      // 4 个 warp 计算 128 维；8 个 warp / block 计算 256；16 个 warp / block 计算 512 维
	const MAT_PTR_TYPE* __restrict__    d_group_offset,
	const MAT_PTR_TYPE* __restrict__    d_tc_offset,
	const MAT_PTR_TYPE* __restrict__    d_row_indices,
	const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
	const vint*         __restrict__    d_sparseA2B,
	const MAT_VAL_TYPE* __restrict__    d_valueA,
	const MAT_VAL_TYPE* __restrict__    d_MatB, 
	MAT_VAL_TYPE*                       d_MatC,
	const vint numNodes,
	const vint feature_dim
);

// __host__
void tf32TransposeCompute(
	TCLOCAL_TYPE* d_tcLocalBit,
	MAT_PTR_TYPE* d_sparseA2B,
	MAT_VAL_TYPE* d_dataA, 
	MAT_VAL_TYPE* d_DenseMatB, 
	MAT_VAL_TYPE* d_DenseMatC, 
	vint* d_block2Idx, 
	vint* d_data2Idx, 
	vint numNodes,
	vint numBlocks,
	vint feature_dim,
	dim3 grid_size, 
	dim3 block_size
);

// __host__
void tf32TransposeAdpBalanceCompute(
	MAT_PTR_TYPE* d_adp_group_offset, 
	MAT_PTR_TYPE* d_tc_offset, 
	MAT_PTR_TYPE* d_adp_row_indices, 
	TCLOCAL_TYPE* d_tcLocalBit, 
	MAT_PTR_TYPE* d_sparseA2B, 
	MAT_VAL_TYPE* d_dataA, 
	MAT_VAL_TYPE* d_DenseMatB, 
	MAT_VAL_TYPE* d_DenseMatC,
	AdpBME<MAT_VAL_TYPE>& adpbme,
	vint numBlocks,
	vint numNodes, 
	vint feature_dim,
	vint denseCSize,
	dim3 grid_size, 
	dim3 block_size
);

/************************************************************************/

#endif // ACC_SPMM_V2_H
