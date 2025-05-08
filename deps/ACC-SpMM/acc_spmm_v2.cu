#include "acc_spmm_v2.h"

/************************************************************************/
/* FUNCTION DEFINITIONS */

__device__ __forceinline__
void tf32_m16n8k8(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
	vint const* A   = reinterpret_cast<vint const*>(MatA);
	vint const* B   = reinterpret_cast<vint const*>(MatB);
	float* C        = reinterpret_cast<float*>(MatC);

	asm volatile(
		"cvt.rna.tf32.f32 %4, %4;\n"
		"cvt.rna.tf32.f32 %5, %5;\n"
		"cvt.rna.tf32.f32 %6, %6;\n"
		"cvt.rna.tf32.f32 %7, %7;\n"
		"cvt.rna.tf32.f32 %8, %8;\n"
		"cvt.rna.tf32.f32 %9, %9;\n"
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
		"{%0, %1, %2, %3},"
		"{%4, %5, %6, %7},"
		"{%8, %9},"
		"{%0, %1, %2, %3};\n"
		:"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
		:"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
		 "r"(B[0]), "r"(B[1])
	);
}

__device__ __forceinline__
void tf32_m16n8k8_detail(
	MAT_VAL_TYPE MatA0, 
	MAT_VAL_TYPE MatA1, 
	MAT_VAL_TYPE MatA2, 
	MAT_VAL_TYPE MatA3, 
	MAT_VAL_TYPE MatB0,
	MAT_VAL_TYPE MatB1, 
	MAT_VAL_TYPE C0,
	MAT_VAL_TYPE C1,
	MAT_VAL_TYPE C2,
	MAT_VAL_TYPE C3
) {
	vint const A0   = static_cast<vint const>(MatA0);
	vint const A1   = static_cast<vint const>(MatA1);
	vint const A2   = static_cast<vint const>(MatA2);
	vint const A3   = static_cast<vint const>(MatA3);
	vint const B0   = static_cast<vint const>(MatB0);
	vint const B1   = static_cast<vint const>(MatB1);
	// float C0        = static_cast<float>(MatC0);
	// float C1        = static_cast<float>(MatC1);
	// float C2        = static_cast<float>(MatC2);
	// float C3        = static_cast<float>(MatC3);

	asm volatile(
		"cvt.rna.tf32.f32 %4, %4;\n"
		"cvt.rna.tf32.f32 %5, %5;\n"
		"cvt.rna.tf32.f32 %6, %6;\n"
		"cvt.rna.tf32.f32 %7, %7;\n"
		"cvt.rna.tf32.f32 %8, %8;\n"
		"cvt.rna.tf32.f32 %9, %9;\n"
		"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
		"{%0, %1, %2, %3},"
		"{%4, %5, %6, %7},"
		"{%8, %9},"
		"{%0, %1, %2, %3};\n"
		:"+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3)      // output
		:"r"(A0), "r"(A1), "r"(A2), "r"(A3),
		 "r"(B0), "r"(B1)
	);
}

__device__ __forceinline__
void tf32_m16n8k4(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
	vint const* A   = reinterpret_cast<vint const*>(MatA);
	vint const* B   = reinterpret_cast<vint const*>(MatB);
	float *C        = reinterpret_cast<float*>(MatC);

	asm volatile(
		"cvt.rna.tf32.f32 %4, %4;\n"
		"cvt.rna.tf32.f32 %5, %5;\n"
		"cvt.rna.tf32.f32 %6, %6;\n"
		"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"
		"{%0, %1, %2, %3},"
		"{%4, %5},"
		"{%6},"
		"{%0, %1, %2, %3};\n"
		:"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
		:"r"(A[0]), "r"(A[1]),
		 "r"(B[0])
	);
}

__device__ __forceinline__
void wait_group() {
	asm volatile(
		"cp.async.commit_group;\n"
		"cp.async.wait_group 0;\n"
		::
	);
}

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
// vint warp_id() {
// 	return threadIdx.x >> 5;
// }

__device__ __forceinline__
void async_copy (MAT_PTR_TYPE shared_addr, const MAT_VAL_TYPE* val) {
	asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__
void async_copy_idx (MAT_PTR_TYPE shared_addr, const vint* val) {
	asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
	// asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared_addr), "l"(val));
}

// Cache in L1, L2.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global(const MAT_VAL_TYPE* a) {
	MAT_VAL_TYPE r;
	asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(r) : "l"(a));
	return r;
}

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global_cs(const MAT_VAL_TYPE* a) {
	MAT_VAL_TYPE r;
	asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(r) : "l"(a));
	return r;
}

// Don't cache and fetch again.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global2shared(const MAT_VAL_TYPE* a) {
	MAT_VAL_TYPE r;
	asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(r) : "l"(a));
	return r;
}

__device__ __forceinline__ 
vint load_int_from_global(const vint* a) {
	int r;
	asm volatile("ld.global.cv.s32 %0, [%1];" : "=r"(r) : "l"(a));      // s32:signed integer
	return r;
}

// recently least used cache friendly
__device__ __forceinline__
void store_fp32_to_global(MAT_VAL_TYPE* a, MAT_VAL_TYPE v) {
	asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(a), "f"(v));
}

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_shared1(const MAT_PTR_TYPE a) {
	MAT_VAL_TYPE r;
	asm volatile("ld.shared.cs.f32 %0, [%1];" : "=f"(r) : "r"(a));
	return r;
}

__device__ __forceinline__
float4 vector_fetch_fp32V4(const float4 *ptr) {
	float4 ret;
	asm volatile (
		"ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
		: "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
		: "l"(ptr)
	);
	return ret;
}

__device__ __forceinline__
float2 vector_fetch_fp32V2(const float2 *ptr) {
	float2 ret;
	asm volatile (
		"ld.global.v2.f32 {%0, %1}, [%2];"
		: "=f"(ret.x), "=f"(ret.y)
		: "l"(ptr)
	);
	return ret;
}

__device__ __forceinline__
MAT_VAL_TYPE load_int_from_shared(const MAT_PTR_TYPE a) {
	vint r;
	asm volatile("ld.shared.cs.s32 %0, [%1];" : "=r"(r) : "r"(a));
	return r;
}

__device__ __forceinline__
float2 ld_shared_float2(uint a) {
	float2 v;
	asm volatile ("ld.shared.v2.f32 {%0, %1}, [%2];"  : "=f"(v.x),"=f"(v.y) : "r"(a*4));
	return v;
}

__device__ __forceinline__
float4 ld_shared_float4(uint a) {
	float4 v;
	asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"  : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(a*4));
	return v;
}

__device__ __forceinline__ 
uint getSMId() {
	uint smid;
	asm("mov.u32 %0, %smid;" : "=r"(smid));
	return smid;
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2];
	using BRegisters = MAT_VAL_TYPE[2][4];
	using CRegisters = MAT_VAL_TYPE[2][4];
	ARegisters fragA;
	BRegisters fragB;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint tid            =   (threadIdx.y << 5) + laneid;  // local thread ID
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint col_off         =   (local_warpID << 5);
	vint colB02          =   groupID + col_off;
	vint colB13          =   groupID + 8 + col_off;
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[2 * idx_len];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
	MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];
	vint start_dataIdx          = d_data2Idx[start_blk_idx];

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		if(present_local & (1ULL << tid))
			// local_idx = __popcll(present_local & ((1ULL << (tid + 1)) - 1));
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		// d_sharedSparseA2B[tid] = d_sparseA2B[start_blk_idx * inst_k + tid];
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
	}
	__syncthreads();

	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint shared_mem_sel      = (tc_block - start_blk_idx + 1) & 1;
		vint shared_mem_sel_next = (tc_block - start_blk_idx) & 1;

		// 取 B
			vint dense_rowIdx01 = d_sharedSparseA2B[(shared_mem_sel << 3) + row01];
			vint dense_rowIdx23 = d_sharedSparseA2B[(shared_mem_sel << 3) + row23];
			
			if(dense_rowIdx01 > numNodes) {
				fragB[0][0] = 0.0; fragB[0][1] = 0.0; 
				fragB[1][0] = 0.0; fragB[1][1] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
				fragB[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB[1][0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB[1][1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
			if(dense_rowIdx23 > numNodes) {
				fragB[0][2] = 0.0; fragB[0][3] = 0.0; 
				fragB[1][2] = 0.0; fragB[1][3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
				fragB[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB[1][2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB[1][3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			} 
		// }

		/* === START ASYNC COPY === */
		start_dataIdx = d_data2Idx[tc_block];
		local_idx = 0;
		// end_dataIdx = d_data2Idx[tc_block + 1];
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			if(present_local & (1ULL << tid))
				// local_idx = __popcll(present_local & ((1ULL << (tid + 1)) - 1));
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[(shared_mem_sel_next << 6) + tid] = 0.0;
			// else d_sharedSparseA[(shared_mem_sel_next << 6) + tid] = d_valueA[start_dataIdx + local_idx - 1];
			else async_copy(saPtr + (((shared_mem_sel_next << 6) + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			async_copy_idx(siPtr + (((shared_mem_sel_next << 3) + tid) << 2), d_sparseA2B + tc_block * inst_k + tid);
		}
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA1];
		
		tf32_m16n8k8(fragB[0], fragA, fragC[0]);
		tf32_m16n8k8(fragB[1], fragA, fragC[1]);

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = (end_blk_idx - start_blk_idx + 1) & 1;
	fragA[0] = d_sharedSparseA[(smem_sel << 6) + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[(smem_sel << 6) + rowA * inst_n + colA1];

	// if(local_warpID < dimTileNum) {
		vint dense_rowIdx01 = d_sharedSparseA2B[(smem_sel << 3) + row01];
		vint dense_rowIdx23 = d_sharedSparseA2B[(smem_sel << 3) + row23];
		if(dense_rowIdx01 > numNodes) {
			fragB[0][0] = 0.0; fragB[0][1] = 0.0; 
			fragB[1][0] = 0.0; fragB[1][1] = 0.0; 
		} else {
			vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
			vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
			fragB[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
			fragB[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
			fragB[1][0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
			fragB[1][1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
		}
		if(dense_rowIdx23 > numNodes) {
			fragB[0][2] = 0.0; fragB[0][3] = 0.0;
			fragB[1][2] = 0.0; fragB[1][3] = 0.0;
		} else {
			vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
			vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
			fragB[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
			fragB[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
			fragB[1][2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
			fragB[1][3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
		}

	tf32_m16n8k8(fragB[0], fragA, fragC[0]);
	tf32_m16n8k8(fragB[1], fragA, fragC[1]);

	vint colC  =  0;
	vint rowC  =  0;
	
	vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY;
	#pragma unroll
	for(vint i = 0; i < 4; ++i) {
		rowC = (tID_in_group << 1) + (i & 0x1);
		if(i < 2) colC = groupID;
		else colC = groupID + 8;
		store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
		store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
	}
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2]; 
	using BRegisters = MAT_VAL_TYPE[4]; 
	using CRegisters = MAT_VAL_TYPE[2][4];
	ARegisters fragA;
	BRegisters fragB00;
	BRegisters fragB01;
	BRegisters fragB10;
	BRegisters fragB11;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint warpSize       =   32;
	const vint tid            =   threadIdx.y * warpSize + laneid; 
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   groupID + (local_warpID << 5);
	vint colB13          =   groupID + 8 + (local_warpID << 5);
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[2 * idx_len];
	// MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
	MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];

	// const vint denseBound = numNodes * feature_dim;

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		vint start_dataIdx         = d_data2Idx[start_blk_idx];
		if(present_local & (1ULL << tid))
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();

	vint dense_rowIdx01 = d_sparseA2B[row01];
	vint dense_rowIdx23 = d_sparseA2B[row23];
	if(dense_rowIdx01 > numNodes) {
		fragB00[0] = 0.0; fragB00[1] = 0.0; 
		fragB01[0] = 0.0; fragB01[1] = 0.0;
	} else {
		vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
		vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
		fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
		fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
		fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
		fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
	}
	if(dense_rowIdx23 > numNodes) {
		fragB00[2] = 0.0; fragB00[3] = 0.0; 
		fragB01[2] = 0.0; fragB01[3] = 0.0;
	} else {
		vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
		vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
		fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
		fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
		fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
		fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
	} 
	__syncthreads();

	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 0->iter1
		vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;  // 1->iter1
		vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
		vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;

	// if(local_warpID < dimTileNum) { 
		vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
		vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
		if(sel_shm_next) {
			if(dense_rowIdx101 > numNodes) {
				fragB10[0] = 0.0; fragB10[1] = 0.0; 
				fragB11[0] = 0.0; fragB11[1] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
				fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
			if(dense_rowIdx123 > numNodes) {
				fragB10[2] = 0.0; fragB10[3] = 0.0; 
				fragB11[2] = 0.0; fragB11[3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
				fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
		} else {
			if(dense_rowIdx101 > numNodes) {
				fragB00[0] = 0.0; fragB00[1] = 0.0; 
				fragB01[0] = 0.0; fragB01[1] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
				fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
			if(dense_rowIdx123 > numNodes) {
				fragB00[2] = 0.0; fragB00[3] = 0.0; 
				fragB01[2] = 0.0; fragB01[3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
				fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
		}         

		/* === START ASYNC COPY === */
		local_idx = 0;
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			vint         start_dataIdx = d_data2Idx[tc_block];
			if(present_local & (1ULL << tid))
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
			else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			if(tc_block + 1 < end_blk_idx)
				async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

		if(sel_shm_next) {
			tf32_m16n8k8(fragB00, fragA, fragC[0]);
			tf32_m16n8k8(fragB01, fragA, fragC[1]);
		} else {
			tf32_m16n8k8(fragB10, fragA, fragC[0]);
			tf32_m16n8k8(fragB11, fragA, fragC[1]);
		}

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

	if(!smem_sel) {
		tf32_m16n8k8(fragB00, fragA, fragC[0]);
		tf32_m16n8k8(fragB01, fragA, fragC[1]);
	} else {
		tf32_m16n8k8(fragB10, fragA, fragC[0]);
		tf32_m16n8k8(fragB11, fragA, fragC[1]);
	}
	
	vint colC  =  0;
	vint rowC  =  0;
	
	vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY;
	#pragma unroll
	for(vint i = 0; i < 4; ++i) {
		rowC = (tID_in_group << 1) + (i & 0x1);
		if(i < 2) colC = groupID;
		else colC = groupID + 8;
		store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
		store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
	}
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2];
	using BRegisters = MAT_VAL_TYPE[4];
	using CRegisters = MAT_VAL_TYPE[2][4];
	ARegisters fragA;
	BRegisters fragB00;
	BRegisters fragB01;
	BRegisters fragB10;
	BRegisters fragB11;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint warpSize       =   32;
	const vint tid            =   threadIdx.y * warpSize + laneid;
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   groupID + (local_warpID << 5);
	vint colB13          =   groupID + 8 + (local_warpID << 5);
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;

	vint colC  =  0;
	vint rowC  =  0;
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[2 * idx_len];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
	MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
	MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
	MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];
	vint         start_dataIdx  = d_tc_offset[start_blk_idx];

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		if(present_local & (1ULL << tid))
			local_idx = __popcll(present_local << (63 - tid));

		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}

	// prefetch A2B idx
	if(tid < inst_k) {
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();

	vint dense_rowIdx01 = d_sparseA2B[row01];
	vint dense_rowIdx23 = d_sparseA2B[row23];
	if(dense_rowIdx01 > numNodes) {
		fragB00[0] = 0.0; fragB00[1] = 0.0; 
		fragB01[0] = 0.0; fragB01[1] = 0.0;
	} else {
		vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
		vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
		fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
		fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
		fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
		fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
	}
	if(dense_rowIdx23 > numNodes) {
		fragB00[2] = 0.0; fragB00[3] = 0.0; 
		fragB01[2] = 0.0; fragB01[3] = 0.0;
	} else {
		vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
		vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
		fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
		fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
		fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
		fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
	} 
	__syncthreads();

	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 0->iter1
		vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;  // 1->iter1
		vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
		vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;

		vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
		vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
		if(sel_shm_next) {
			if(dense_rowIdx101 > numNodes) {
				fragB10[0] = 0.0; fragB10[1] = 0.0; 
				fragB11[0] = 0.0; fragB11[1] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
				fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
			if(dense_rowIdx123 > numNodes) {
				fragB10[2] = 0.0; fragB10[3] = 0.0; 
				fragB11[2] = 0.0; fragB11[3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
				fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
		} else {
			if(dense_rowIdx101 > numNodes) {
				fragB00[0] = 0.0; fragB00[1] = 0.0; 
				fragB01[0] = 0.0; fragB01[1] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
				fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
			if(dense_rowIdx123 > numNodes) {
				fragB00[2] = 0.0; fragB00[3] = 0.0; 
				fragB01[2] = 0.0; fragB01[3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
				fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
				fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
			}
		}         

		/* === START ASYNC COPY === */
		local_idx = 0;
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			start_dataIdx = d_tc_offset[tc_block];
			if(present_local & (1ULL << tid))
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
			else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			if(tc_block + 1 < end_blk_idx)
				async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

		if(sel_shm_next) {
			tf32_m16n8k8(fragB00, fragA, fragC[0]);
			tf32_m16n8k8(fragB01, fragA, fragC[1]);
		} else {
			tf32_m16n8k8(fragB10, fragA, fragC[0]);
			tf32_m16n8k8(fragB11, fragA, fragC[1]);
		}

		next_row_idx = d_row_indices[tc_block];

		if(next_row_idx != start_row_idx) {
			vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
			#pragma unroll
			for(vint i = 0; i < 4; ++i) {
				rowC = (tID_in_group << 1) + (i & 0x1);
				if(i < 2) colC = groupID;
				else colC = groupID + 8;
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
				fragC[0][i] = 0.0;
				fragC[1][i] = 0.0;
			}
		}
		start_row_idx = next_row_idx;

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

	if(!smem_sel) {
		tf32_m16n8k8(fragB00, fragA, fragC[0]);
		tf32_m16n8k8(fragB01, fragA, fragC[1]);
	} else {
		tf32_m16n8k8(fragB10, fragA, fragC[0]);
		tf32_m16n8k8(fragB11, fragA, fragC[1]);
	}
	
	vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
	#pragma unroll
	for(vint i = 0; i < 4; ++i) {
		rowC = (tID_in_group << 1) + (i & 0x1);
		if(i < 2) colC = groupID;
		else colC = groupID + 8;
		atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
		atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
	}
}

// #define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
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
) {
	using ARegisters = MAT_VAL_TYPE[2];
	using BRegisters = MAT_VAL_TYPE[4];
	using CRegisters = MAT_VAL_TYPE[4][4];
	ARegisters fragA;
	BRegisters fragB00;
	BRegisters fragB01;
	BRegisters fragB02;
	BRegisters fragB03;
	BRegisters fragB10;
	BRegisters fragB11;
	BRegisters fragB12;
	BRegisters fragB13;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint dimTileNum     =   feature_dim / (COL_WINDOW_R << 2);
	const vint tid            =   (threadIdx.y << 5) + laneid;
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   (groupID << 2) + (local_warpID << 6);
	vint colB13          =   (groupID << 2) + (local_warpID << 6) + 8;
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	vint colC  =  0;
	vint rowC  =  0;

	vint denseBound = numNodes * feature_dim;
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[2 * idx_len];
	// MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
	MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];

	// const vint denseBound = numNodes * feature_dim;

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		vint start_dataIdx         = d_data2Idx[start_blk_idx];
		if(present_local & (1ULL << tid))
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();
	if(local_warpID < dimTileNum) {
		vint dense_rowIdx01 = d_sharedSparseA2B[row01];
		vint dense_rowIdx23 = d_sharedSparseA2B[row23];
		if(dense_rowIdx01 > numNodes) {
			fragB00[0] = 0.0; 
			fragB01[0] = 0.0; 
			fragB02[0] = 0.0;  
			fragB03[0] = 0.0; 

			fragB00[1] = 0.0; 
			fragB01[1] = 0.0;
			fragB02[1] = 0.0;
			fragB03[1] = 0.0;
		} else{
			vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
			vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
			float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
			float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
			fragB00[0] = t0.x;
			fragB01[0] = t0.y;
			fragB02[0] = t0.w;
			fragB03[0] = t0.z;

			fragB00[1] = t1.x;
			fragB01[1] = t1.y;
			fragB02[1] = t1.w;
			fragB03[1] = t1.z;
		}
		if(dense_rowIdx23 > numNodes) {
			fragB00[2] = 0.0;  
			fragB01[2] = 0.0; 
			fragB02[2] = 0.0;  
			fragB03[2] = 0.0; 

			fragB00[3] = 0.0;
			fragB01[3] = 0.0;
			fragB02[3] = 0.0;
			fragB03[3] = 0.0;
		} else{
			vint sourceIdx2 = dense_rowIdx23 * feature_dim + colB02;
			vint sourceIdx3 = dense_rowIdx23 * feature_dim + colB13;
			float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
			float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
			fragB00[2] = t2.x;
			fragB01[2] = t2.y;
			fragB02[2] = t2.w;
			fragB03[2] = t2.z;

			fragB00[3] = t3.x;
			fragB01[3] = t3.y;
			fragB02[3] = t3.w;
			fragB03[3] = t3.z;
		}
	}
	__syncthreads();
	
	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
		vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;
		vint sel_shm           =   sel_idx_shm << 3;
		vint sel_shm_next      =   sel_idx_shm_next << 3; 

		// 取 B
		if(local_warpID < dimTileNum) {
			vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
			vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
			if(sel_shm_next) {
			// if(local_warpID < dimTileNum) {
				if(dense_rowIdx101 > numNodes) {
					fragB10[0] = 0.0; 
					fragB11[0] = 0.0; 
					fragB12[0] = 0.0;  
					fragB13[0] = 0.0; 

					fragB10[1] = 0.0; 
					fragB11[1] = 0.0;
					fragB12[1] = 0.0;
					fragB13[1] = 0.0;
				} else{
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
					float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
					fragB10[0] = t0.x;
					fragB11[0] = t0.y;
					fragB12[0] = t0.w;
					fragB13[0] = t0.z;

					fragB10[1] = t1.x;
					fragB11[1] = t1.y;
					fragB12[1] = t1.w;
					fragB13[1] = t1.z;
				}
				if(dense_rowIdx123 > numNodes) {
					fragB10[2] = 0.0;  
					fragB11[2] = 0.0; 
					fragB12[2] = 0.0;  
					fragB13[2] = 0.0; 

					fragB10[3] = 0.0;
					fragB11[3] = 0.0;
					fragB12[3] = 0.0;
					fragB13[3] = 0.0;
				} else{
					vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
					float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
					float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
					fragB10[2] = t2.x;
					fragB11[2] = t2.y;
					fragB12[2] = t2.w;
					fragB13[2] = t2.z;

					fragB10[3] = t3.x;
					fragB11[3] = t3.y;
					fragB12[3] = t3.w;
					fragB13[3] = t3.z;
				}
			} else {
				if(dense_rowIdx101 > numNodes) {
					fragB00[0] = 0.0; 
					fragB01[0] = 0.0; 
					fragB02[0] = 0.0;  
					fragB03[0] = 0.0; 

					fragB00[1] = 0.0; 
					fragB01[1] = 0.0;
					fragB02[1] = 0.0;
					fragB03[1] = 0.0;
				} else{
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
					float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
					fragB00[0] = t0.x;
					fragB01[0] = t0.y;
					fragB02[0] = t0.w;
					fragB03[0] = t0.z;

					fragB00[1] = t1.x;
					fragB01[1] = t1.y;
					fragB02[1] = t1.w;
					fragB03[1] = t1.z;
				}
				if(dense_rowIdx123 > numNodes) {
					fragB00[2] = 0.0;  
					fragB01[2] = 0.0; 
					fragB02[2] = 0.0;  
					fragB03[2] = 0.0; 

					fragB00[3] = 0.0;
					fragB01[3] = 0.0;
					fragB02[3] = 0.0;
					fragB03[3] = 0.0;
				} else {
					vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
					float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
					float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
					fragB00[2] = t2.x;
					fragB01[2] = t2.y;
					fragB02[2] = t2.w;
					fragB03[2] = t2.z;

					fragB00[3] = t3.x;
					fragB01[3] = t3.y;
					fragB02[3] = t3.w;
					fragB03[3] = t3.z;
				}
			}
		}
		
		/* === START ASYNC COPY === */
		local_idx = 0;
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			vint         start_dataIdx = d_data2Idx[tc_block];
			if(present_local & (1ULL << tid))
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
			else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			if(tc_block + 1 < end_blk_idx)
				async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

		if(sel_shm_next) {
			tf32_m16n8k8(fragB00, fragA, fragC[0]);
			tf32_m16n8k8(fragB01, fragA, fragC[1]);
			tf32_m16n8k8(fragB02, fragA, fragC[2]);
			tf32_m16n8k8(fragB03, fragA, fragC[3]);
		} else {
			tf32_m16n8k8(fragB10, fragA, fragC[0]);
			tf32_m16n8k8(fragB11, fragA, fragC[1]);
			tf32_m16n8k8(fragB12, fragA, fragC[2]);
			tf32_m16n8k8(fragB13, fragA, fragC[3]);
		}

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

	if(!smem_sel) {
		tf32_m16n8k8(fragB00, fragA, fragC[0]);
		tf32_m16n8k8(fragB01, fragA, fragC[1]);
		tf32_m16n8k8(fragB02, fragA, fragC[2]);
		tf32_m16n8k8(fragB03, fragA, fragC[3]);
	} else {
		tf32_m16n8k8(fragB10, fragA, fragC[0]);
		tf32_m16n8k8(fragB11, fragA, fragC[1]);
		tf32_m16n8k8(fragB12, fragA, fragC[2]);
		tf32_m16n8k8(fragB13, fragA, fragC[3]);
	}
	
	if(local_warpID < dimTileNum) {
		vint outOff = (bid << 3) * feature_dim + (local_warpID << 6) + offY;
		#pragma unroll
		for(vint i = 0; i < 4; ++i) {
			rowC = (tID_in_group << 1) + (i & 0x1);
			if(i < 2) colC = groupID << 2;
			else colC = (groupID + 8) << 2;
			vint off = outOff + rowC * feature_dim + colC;
			store_fp32_to_global(d_MatC + off, fragC[0][i]);
			store_fp32_to_global(d_MatC + off + 1, fragC[1][i]);
			store_fp32_to_global(d_MatC + off + 2, fragC[2][i]);
			store_fp32_to_global(d_MatC + off + 3, fragC[3][i]);
		}
	}
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2];
	using BRegisters = MAT_VAL_TYPE[4]; 
	using CRegisters = MAT_VAL_TYPE[4][4]; 
	ARegisters fragA;
	BRegisters fragB00;
	BRegisters fragB01;
	BRegisters fragB02;
	BRegisters fragB03;
	BRegisters fragB10;
	BRegisters fragB11;
	BRegisters fragB12;
	BRegisters fragB13;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint dimTileNum     =   feature_dim / (COL_WINDOW_R << 2);
	const vint tid            =   (threadIdx.y << 5) + laneid;
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   (groupID << 2) + (local_warpID << 6);
	vint colB13          =   (groupID << 2) + (local_warpID << 6) + 8;
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	vint colC  =  0;
	vint rowC  =  0;

	vint denseBound = numNodes * feature_dim;
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[2 * idx_len];
	// MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
	MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
	MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
	MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];

	// const vint denseBound = numNodes * feature_dim;

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		vint         start_dataIdx = d_tc_offset[start_blk_idx];
		if(present_local & (1ULL << tid))
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		// d_sharedSparseA2B[tid] = d_sparseA2B[start_blk_idx * inst_k + tid];
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();
	if(local_warpID < dimTileNum) {
		vint dense_rowIdx01 = d_sharedSparseA2B[row01];
		vint dense_rowIdx23 = d_sharedSparseA2B[row23];
		if(dense_rowIdx01 > numNodes) {
			fragB00[0] = 0.0; 
			fragB01[0] = 0.0; 
			fragB02[0] = 0.0;  
			fragB03[0] = 0.0; 

			fragB00[1] = 0.0; 
			fragB01[1] = 0.0;
			fragB02[1] = 0.0;
			fragB03[1] = 0.0;
		} else{
			vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
			vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
			// float4 t0 = FLOAT4(d_MatB[sourceIdx0]);
			// float4 t1 = FLOAT4(d_MatB[sourceIdx1]);
			float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
			float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
			fragB00[0] = t0.x;
			fragB01[0] = t0.y;
			fragB02[0] = t0.w;
			fragB03[0] = t0.z;

			fragB00[1] = t1.x;
			fragB01[1] = t1.y;
			fragB02[1] = t1.w;
			fragB03[1] = t1.z;
		}
		if(dense_rowIdx23 > numNodes) {
			fragB00[2] = 0.0;  
			fragB01[2] = 0.0; 
			fragB02[2] = 0.0;  
			fragB03[2] = 0.0; 

			fragB00[3] = 0.0;
			fragB01[3] = 0.0;
			fragB02[3] = 0.0;
			fragB03[3] = 0.0;
		} else{
			vint sourceIdx2 = dense_rowIdx23 * feature_dim + colB02;
			vint sourceIdx3 = dense_rowIdx23 * feature_dim + colB13;
			// float4 t2 = FLOAT4(d_MatB[sourceIdx2]);
			// float4 t3 = FLOAT4(d_MatB[sourceIdx3]);
			float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
			float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
			fragB00[2] = t2.x;
			fragB01[2] = t2.y;
			fragB02[2] = t2.w;
			fragB03[2] = t2.z;

			fragB00[3] = t3.x;
			fragB01[3] = t3.y;
			fragB02[3] = t3.w;
			fragB03[3] = t3.z;
		}
	}
	__syncthreads();
	
	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
		vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;
		vint sel_shm           =   sel_idx_shm << 3;   // 0->iter1
		vint sel_shm_next      =   sel_idx_shm_next << 3;  // 1->iter1

		// 取 B
		if(local_warpID < dimTileNum) {
			vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
			vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
			// float4 t0   = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
			// float4 t1   = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
			if(sel_shm_next) {
			// if(local_warpID < dimTileNum) {
				if(dense_rowIdx101 > numNodes) {
					fragB10[0] = 0.0; 
					fragB11[0] = 0.0; 
					fragB12[0] = 0.0;  
					fragB13[0] = 0.0; 

					fragB10[1] = 0.0; 
					fragB11[1] = 0.0;
					fragB12[1] = 0.0;
					fragB13[1] = 0.0;
				} else{
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					// float4 t0 = FLOAT4(d_MatB[sourceIdx0]);
					// float4 t1 = FLOAT4(d_MatB[sourceIdx1]);
					float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
					float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
					fragB10[0] = t0.x;
					fragB11[0] = t0.y;
					fragB12[0] = t0.w;
					fragB13[0] = t0.z;

					fragB10[1] = t1.x;
					fragB11[1] = t1.y;
					fragB12[1] = t1.w;
					fragB13[1] = t1.z;
				}
				if(dense_rowIdx123 > numNodes) {
					fragB10[2] = 0.0;  
					fragB11[2] = 0.0; 
					fragB12[2] = 0.0;  
					fragB13[2] = 0.0; 

					fragB10[3] = 0.0;
					fragB11[3] = 0.0;
					fragB12[3] = 0.0;
					fragB13[3] = 0.0;
				} else{
					vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
					float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
					float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
					fragB10[2] = t2.x;
					fragB11[2] = t2.y;
					fragB12[2] = t2.w;
					fragB13[2] = t2.z;

					fragB10[3] = t3.x;
					fragB11[3] = t3.y;
					fragB12[3] = t3.w;
					fragB13[3] = t3.z;
				}
			} else {
				if(dense_rowIdx101 > numNodes) {
					fragB00[0] = 0.0; 
					fragB01[0] = 0.0; 
					fragB02[0] = 0.0;  
					fragB03[0] = 0.0; 

					fragB00[1] = 0.0; 
					fragB01[1] = 0.0;
					fragB02[1] = 0.0;
					fragB03[1] = 0.0;
				} else{
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
					float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
					fragB00[0] = t0.x;
					fragB01[0] = t0.y;
					fragB02[0] = t0.w;
					fragB03[0] = t0.z;

					fragB00[1] = t1.x;
					fragB01[1] = t1.y;
					fragB02[1] = t1.w;
					fragB03[1] = t1.z;
				}
				if(dense_rowIdx123 > numNodes) {
					fragB00[2] = 0.0;  
					fragB01[2] = 0.0; 
					fragB02[2] = 0.0;  
					fragB03[2] = 0.0; 

					fragB00[3] = 0.0;
					fragB01[3] = 0.0;
					fragB02[3] = 0.0;
					fragB03[3] = 0.0;
				} else {
					vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
					float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
					float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
					fragB00[2] = t2.x;
					fragB01[2] = t2.y;
					fragB02[2] = t2.w;
					fragB03[2] = t2.z;

					fragB00[3] = t3.x;
					fragB01[3] = t3.y;
					fragB02[3] = t3.w;
					fragB03[3] = t3.z;
				}
			}
		}
		
		/* === START ASYNC COPY === */
		local_idx = 0;
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			vint         start_dataIdx = d_tc_offset[tc_block];
			if(present_local & (1ULL << tid))
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
			else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			if(tc_block + 1 < end_blk_idx)
				async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

		if(sel_shm_next) {
			tf32_m16n8k8(fragB00, fragA, fragC[0]);
			tf32_m16n8k8(fragB01, fragA, fragC[1]);
			tf32_m16n8k8(fragB02, fragA, fragC[2]);
			tf32_m16n8k8(fragB03, fragA, fragC[3]);
		} else {
			tf32_m16n8k8(fragB10, fragA, fragC[0]);
			tf32_m16n8k8(fragB11, fragA, fragC[1]);
			tf32_m16n8k8(fragB12, fragA, fragC[2]);
			tf32_m16n8k8(fragB13, fragA, fragC[3]);
		}

		next_row_idx = d_row_indices[tc_block];

		if(next_row_idx != start_row_idx) {
			vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
			#pragma unroll
			for(vint i = 0; i < 4; ++i) {
				rowC = (tID_in_group << 1) + (i & 0x1);
				if(i < 2) colC = groupID << 2;
				else colC = (groupID + 8) << 2;
				vint off = outOff + rowC * feature_dim + colC;
				atomicAdd(d_MatC + off, fragC[0][i]);
				atomicAdd(d_MatC + off + 1, fragC[1][i]);
				atomicAdd(d_MatC + off + 2, fragC[2][i]);
				atomicAdd(d_MatC + off + 3, fragC[3][i]);
			}
		}
		start_row_idx = next_row_idx;

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

	if(!smem_sel) {
		tf32_m16n8k8(fragB00, fragA, fragC[0]);
		tf32_m16n8k8(fragB01, fragA, fragC[1]);
		tf32_m16n8k8(fragB02, fragA, fragC[2]);
		tf32_m16n8k8(fragB03, fragA, fragC[3]);
	} else {
		tf32_m16n8k8(fragB10, fragA, fragC[0]);
		tf32_m16n8k8(fragB11, fragA, fragC[1]);
		tf32_m16n8k8(fragB12, fragA, fragC[2]);
		tf32_m16n8k8(fragB13, fragA, fragC[3]);
	}
	
	if(local_warpID < dimTileNum) {
		vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
		#pragma unroll
		for(vint i = 0; i < 4; ++i) {
			rowC = (tID_in_group << 1) + (i & 0x1);
			if(i < 2) colC = groupID << 2;
			else colC = (groupID + 8) << 2;
			vint off = outOff + rowC * feature_dim + colC;
			atomicAdd(d_MatC + off, fragC[0][i]);
			atomicAdd(d_MatC + off + 1, fragC[1][i]);
			atomicAdd(d_MatC + off + 2, fragC[2][i]);
			atomicAdd(d_MatC + off + 3, fragC[3][i]);
		}
	}
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
	using BRegisters = MAT_VAL_TYPE[2][4];     // 算2个 m16n8k8，共用一个 A 16 * 8
	using CRegisters = MAT_VAL_TYPE[2][4];  // 16 * 8
	ARegisters fragA;
	BRegisters fragB0;
	BRegisters fragB1;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	// vint out_RowOff           =   (blockIdx.x << 4);    // * ROW_WINDOW
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint warpSize       =   32;
	// const vint threadPerBlk   =   blockDim.x * blockDim.y;
	// const vint dimTileNum     =   feature_dim / (COL_WINDOW_R << 1);
	const vint tid            =   threadIdx.y * warpSize + laneid;  // local thread ID
	// const vint global_warpID  =   (global_tid >> 5) * WARP_NUM;
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	// load A 重新映射，now A is B，映射取数的行和列 8 * 8 part
	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   groupID + (local_warpID << 5);
	vint colB13          =   groupID + 8 + (local_warpID << 5);
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	vint colC  =  0;
	vint rowC  =  0;

	// vint denseDimIdx  = local_warpID * COL_WINDOW_R * 2;      // 一个 warp 算2个 B
	// vint denseDimIdx  = (local_warpID << 5);      // 一个 warp 算2个 B

	// vint dense_rowIdx_off = groupID;     // for fetching idx
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[3 * idx_len];
	// MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
	MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
	MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
	MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];
	vint         start_dataIdx  = d_tc_offset[start_blk_idx];

	// const vint denseBound = numNodes * feature_dim;

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		if(present_local & (1ULL << tid))
			// local_idx = __popcll(present_local & ((1ULL << (tid + 1)) - 1));
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		// d_sharedSparseA2B[tid] = d_sparseA2B[start_blk_idx * inst_k + tid];
		d_sharedSparseA2B[tid]      = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8]  = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();

	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint shared_mem_sel      = ((tc_block - start_blk_idx + 1) & 1) << 6;
		vint shared_mem_sel_next = ((tc_block - start_blk_idx) & 1) << 6;

		vint sel_idx             =  ((tc_block - start_blk_idx - 1) % 3) << 3;
		vint sel_idx_next        =  ((tc_block - start_blk_idx    ) % 3) << 3;
		vint sel_idx_next_next   =  ((tc_block - start_blk_idx + 1) % 3) << 3;

		// 取 B
		if(shared_mem_sel_next) {
			// if(local_warpID < dimTileNum) {
				// vint dense_rowIdx0 = d_sharedSparseA2B[(shared_mem_sel << 3) + dense_rowIdx_off];
				vint dense_rowIdx01 = d_sharedSparseA2B[sel_idx + row01];
				vint dense_rowIdx23 = d_sharedSparseA2B[sel_idx + row23];
				if(dense_rowIdx01 > numNodes) {
					fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
					fragB0[1][0] = 0.0; fragB0[1][1] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
				}
				if(dense_rowIdx23 > numNodes) {
					fragB0[0][2] = 0.0; fragB0[0][3] = 0.0; 
					fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
				} 

				vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_next + row01];
				vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_next + row23];
				if(dense_rowIdx101 > numNodes) {
					fragB1[0][0] = 0.0; fragB1[0][1] = 0.0; 
					fragB1[1][0] = 0.0; fragB1[1][1] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB1[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB1[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB1[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB1[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
				}
				if(dense_rowIdx123 > numNodes) {
					fragB1[0][2] = 0.0; fragB1[0][3] = 0.0; 
					fragB1[1][2] = 0.0; fragB1[1][3] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB1[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB1[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB1[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB1[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
				} 
			// }
		}
		
		/* === START ASYNC COPY === */
		start_dataIdx = d_tc_offset[tc_block];
		local_idx = 0;
		// end_dataIdx = d_data2Idx[tc_block + 1];
		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			if(present_local & (1ULL << tid))
				// local_idx = __popcll(present_local & ((1ULL << (tid + 1)) - 1));
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[shared_mem_sel_next + tid] = 0.0;
			// else d_sharedSparseA[(shared_mem_sel_next << 6) + tid] = d_valueA[start_dataIdx + local_idx - 1];
			else async_copy(saPtr + ((shared_mem_sel_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			// d_sharedSparseA2B[(shared_mem_sel_next << 3) + tid] = d_sparseA2B[tc_block * inst_k + tid];
			if(tc_block + 1 < end_blk_idx)
				async_copy_idx(siPtr + ((sel_idx_next_next + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[shared_mem_sel + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[shared_mem_sel + rowA * inst_n + colA1];
		// fragA[0] = load_fp32_from_shared1(saPtr + (((shared_mem_sel << 6) + rowA * inst_n + colA0) << 2));
		// fragA[1] = load_fp32_from_shared1(saPtr + (((shared_mem_sel << 6) + rowA * inst_n + colA1) << 2));

		if(shared_mem_sel_next) {
			tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
			tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
		} else {
			tf32_m16n8k8(fragB1[0], fragA, fragC[0]);
			tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
		}

		next_row_idx = d_row_indices[tc_block];
		if(next_row_idx != start_row_idx) {
			vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
			#pragma unroll
			for(vint i = 0; i < 4; ++i) {
				rowC = (tID_in_group << 1) + (i & 0x1);
				if(i < 2) colC = groupID;
				else colC = groupID + 8;
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
				fragC[0][i] = 0.0; fragC[1][i] = 0.0;
			}
		}
		start_row_idx = next_row_idx;
		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	vint sel_idx   = ((end_blk_idx - start_blk_idx - 1) % 3) << 3;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];
	// fragA[0] = load_fp32_from_shared1(saPtr + (((smem_sel << 6) + rowA * inst_n + colA0) << 2));
	// fragA[1] = load_fp32_from_shared1(saPtr + (((smem_sel << 6) + rowA * inst_n + colA1) << 2));

	if(!smem_sel) {
		// if(local_warpID < dimTileNum) {
			vint dense_rowIdx01 = d_sharedSparseA2B[sel_idx + row01];
			vint dense_rowIdx23 = d_sharedSparseA2B[sel_idx + row23];
			if(dense_rowIdx01 > numNodes) {
				fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
				fragB0[1][0] = 0.0; fragB0[1][1] = 0.0; 
			} else {
				vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
				vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
				vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
				fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
				fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);

			}
			if(dense_rowIdx23 > numNodes) {
				fragB0[0][2] = 0.0; fragB0[0][3] = 0.0;
				fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
				vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
				vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
				fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
				fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
			}        
		// }
		tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
		tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
	} else {
		tf32_m16n8k8(fragB1[0], fragA, fragC[0]);
		tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
	}
	
	// if(local_warpID < dimTileNum) {
		vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
		#pragma unroll
		for(vint i = 0; i < 4; ++i) {
			rowC = (tID_in_group << 1) + (i & 0x1);
			if(i < 2) colC = groupID;
			else colC = groupID + 8;
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
		}
	// }
}

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
) {
	using ARegisters = MAT_VAL_TYPE[2];     // 8 * 8
	using BRegisters = MAT_VAL_TYPE[4][4];     // 算2个 m16n8k8，共用一个 A 16 * 8
	using CRegisters = MAT_VAL_TYPE[4][4];  // 16 * 8
	ARegisters fragA;
	BRegisters fragB0;
	BRegisters fragB1;
	CRegisters fragC = {0.0};

	vint bid                  =   blockIdx.x;
	// vint out_RowOff           =   (blockIdx.x << 4);    // * ROW_WINDOW
	vint offY                 =   (blockIdx.y << 7);
	const vint laneid         =   31 & threadIdx.x;
	const vint warpSize       =   32;
	// const vint threadPerBlk   =   blockDim.x * blockDim.y;
	// const vint dimTileNum     =   feature_dim / (COL_WINDOW_R << 2);
	const vint tid            =   threadIdx.y * warpSize + laneid;  // local thread ID
	// const vint global_warpID  =   (global_tid >> 5) * WARP_NUM;
	const vint local_warpID   =   threadIdx.y;

	vint groupID         =   laneid >> 2;
	vint tID_in_group    =   3 & laneid;

	// load A 重新映射，now A is B，映射取数的行和列 8 * 8 part
	vint rowA            =   groupID;
	vint colA0           =   tID_in_group;
	vint colA1           =   tID_in_group + 4;

	vint colB02          =   groupID + (local_warpID << 6);
	vint colB13          =   groupID + 8 + (local_warpID << 6);
	vint row01           =   tID_in_group;
	vint row23           =   tID_in_group + 4;
	
	vint colC  =  0;
	vint rowC  =  0;

	// vint denseDimIdx  = local_warpID * COL_WINDOW_R * 2;      // 一个 warp 算2个 B
	// vint denseDimIdx  = (local_warpID << 5);      // 一个 warp 算2个 B

	// vint dense_rowIdx_off = groupID;     // for fetching idx
	
	constexpr const int inst_k  = 8;
	constexpr const int inst_n  = 8;

	const vint mat_len = 64;
	const vint idx_len = 8;
	vint  local_idx    = 0;

	__shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
	__shared__ vint         d_sharedSparseA2B[3 * idx_len];
	// MAT_VAL_TYPE            d_denseB[inst_m * inst_n];
	
	vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
	vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
	
	MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
	MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
	MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
	MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];
	vint         start_dataIdx  = d_tc_offset[start_blk_idx];

	// const vint denseBound = numNodes * feature_dim;

	/* === pre loop === */
	if(tid < mat_len) {  
		TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
		if(present_local & (1ULL << tid))
			// local_idx = __popcll(present_local & ((1ULL << (tid + 1)) - 1));
			local_idx = __popcll(present_local << (63 - tid));
		// prefetch 1 tc_block
		if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
		else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
	}
	// prefetch A2B idx
	if(tid < inst_k) {
		// d_sharedSparseA2B[tid] = d_sparseA2B[start_blk_idx * inst_k + tid];
		d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
		if(start_blk_idx + 1 < end_blk_idx) {
			d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
		}
	}
	__syncthreads();

	/* === main loop === */
	for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
		vint shared_mem_sel      = ((tc_block - start_blk_idx + 1) & 1) << 6;
		vint shared_mem_sel_next = ((tc_block - start_blk_idx) & 1) << 6;

		vint sel_idx             =  ((tc_block - start_blk_idx - 1) % 3) << 3;
		vint sel_idx_next        =  ((tc_block - start_blk_idx    ) % 3) << 3;
		vint sel_idx_next_next   =  ((tc_block - start_blk_idx + 1) % 3) << 3;

		// 取 B
		if(shared_mem_sel_next) {
			// if(local_warpID < dimTileNum) {
				vint dense_rowIdx01 = d_sharedSparseA2B[sel_idx + row01];
				vint dense_rowIdx23 = d_sharedSparseA2B[sel_idx + row23];
				if(dense_rowIdx01 > numNodes) {
					fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
					fragB0[1][0] = 0.0; fragB0[1][1] = 0.0;
					fragB0[2][0] = 0.0; fragB0[2][1] = 0.0; 
					fragB0[3][0] = 0.0; fragB0[3][1] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);

					vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
					vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
					vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
					vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
					fragB0[2][0] = load_fp32_from_global(d_MatB + sourceIdx4); 
					fragB0[2][1] = load_fp32_from_global(d_MatB + sourceIdx5); 
					fragB0[3][0] = load_fp32_from_global(d_MatB + sourceIdx6); 
					fragB0[3][1] = load_fp32_from_global(d_MatB + sourceIdx7); 
				}
				if(dense_rowIdx23 > numNodes) {
					fragB0[0][2] = 0.0; fragB0[0][3] = 0.0; 
					fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
					fragB0[2][2] = 0.0; fragB0[2][3] = 0.0; 
					fragB0[3][2] = 0.0; fragB0[3][3] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);

					vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
					vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
					vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
					vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
					fragB0[2][2] = load_fp32_from_global(d_MatB + sourceIdx4); 
					fragB0[2][3] = load_fp32_from_global(d_MatB + sourceIdx5);
					fragB0[3][2] = load_fp32_from_global(d_MatB + sourceIdx6); 
					fragB0[3][3] = load_fp32_from_global(d_MatB + sourceIdx7);
				} 

				vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_next + row01];
				vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_next + row23];

				if(dense_rowIdx101 > numNodes) {
					fragB1[0][0] = 0.0; fragB1[0][1] = 0.0; 
					fragB1[1][0] = 0.0; fragB1[1][1] = 0.0;
					fragB1[2][0] = 0.0; fragB1[2][1] = 0.0; 
					fragB1[3][0] = 0.0; fragB1[3][1] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB1[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB1[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB1[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB1[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);

					vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
					vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
					vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
					vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
					fragB1[2][0] = load_fp32_from_global(d_MatB + sourceIdx4); 
					fragB1[2][1] = load_fp32_from_global(d_MatB + sourceIdx5); 
					fragB1[3][0] = load_fp32_from_global(d_MatB + sourceIdx6); 
					fragB1[3][1] = load_fp32_from_global(d_MatB + sourceIdx7); 
				}
				if(dense_rowIdx123 > numNodes) {
					fragB1[0][2] = 0.0; fragB1[0][3] = 0.0; 
					fragB1[1][2] = 0.0; fragB1[1][3] = 0.0;
					fragB1[2][2] = 0.0; fragB1[2][3] = 0.0; 
					fragB1[3][2] = 0.0; fragB1[3][3] = 0.0;
				} else {
					vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
					vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
					vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
					vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
					fragB1[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
					fragB1[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
					fragB1[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
					fragB1[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);

					vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
					vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
					vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
					vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
					fragB1[2][2] = load_fp32_from_global(d_MatB + sourceIdx4); 
					fragB1[2][3] = load_fp32_from_global(d_MatB + sourceIdx5);
					fragB1[3][2] = load_fp32_from_global(d_MatB + sourceIdx6); 
					fragB1[3][3] = load_fp32_from_global(d_MatB + sourceIdx7);
				} 
			// }
		}
		
		/* === START ASYNC COPY === */
		start_dataIdx = d_tc_offset[tc_block];
		local_idx = 0;

		if(tid < mat_len) {  
			TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
			if(present_local & (1ULL << tid))
				local_idx = __popcll(present_local << (63 - tid));
			if(local_idx == 0) d_sharedSparseA[shared_mem_sel_next + tid] = 0.0;
			else async_copy(saPtr + ((shared_mem_sel_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
		}
		if(tid < inst_k) {
			async_copy_idx(siPtr + ((sel_idx_next_next + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
		}
		/* === END OF ASYNC COPY === */
		// fetch A
		fragA[0] = d_sharedSparseA[shared_mem_sel + rowA * inst_n + colA0];
		fragA[1] = d_sharedSparseA[shared_mem_sel + rowA * inst_n + colA1];

		if(shared_mem_sel_next) {
			tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
			tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
			tf32_m16n8k8(fragB0[2], fragA, fragC[2]);
			tf32_m16n8k8(fragB0[3], fragA, fragC[3]);
		} else {
			tf32_m16n8k8(fragB1[0], fragA, fragC[0]);
			tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
			tf32_m16n8k8(fragB1[2], fragA, fragC[2]);
			tf32_m16n8k8(fragB1[3], fragA, fragC[3]);
		}

		next_row_idx = d_row_indices[tc_block];

		if(next_row_idx != start_row_idx) {
			// printf("write-back: %d, tc_block: %d\n", start_row_idx, tc_block);
			vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
			#pragma unroll
			for(vint i = 0; i < 4; ++i) {
				rowC = (tID_in_group << 1) + (i & 0x1);
				if(i < 2) colC = groupID;
				else colC = groupID + 8;
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 2, fragC[2][i]);
				atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 3, fragC[3][i]);
				fragC[0][i] = 0.0; fragC[1][i] = 0.0; fragC[2][i] = 0.0; fragC[3][i] = 0.0; 
			}
		}
		start_row_idx = next_row_idx;

		wait_group();
		__syncthreads();
	}

	/* === end loop === */
	vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
	vint sel_idx   = ((end_blk_idx - start_blk_idx - 1) % 3) << 3;
	fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
	fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

	if(!smem_sel) {
		// if(local_warpID < dimTileNum) {
			vint dense_rowIdx01 = d_sharedSparseA2B[sel_idx + row01];
			vint dense_rowIdx23 = d_sharedSparseA2B[sel_idx + row23];
			if(dense_rowIdx01 > numNodes) {
				fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
				fragB0[1][0] = 0.0; fragB0[1][1] = 0.0; 
				fragB0[2][0] = 0.0; fragB0[2][1] = 0.0; 
				fragB0[3][0] = 0.0; fragB0[3][1] = 0.0; 
			} else {
				vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
				vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
				vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
				fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
				fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
				vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
				vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
				vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
				vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
				fragB0[2][0] = load_fp32_from_global(d_MatB + sourceIdx4); 
				fragB0[2][1] = load_fp32_from_global(d_MatB + sourceIdx5); 
				fragB0[3][0] = load_fp32_from_global(d_MatB + sourceIdx6); 
				fragB0[3][1] = load_fp32_from_global(d_MatB + sourceIdx7); 
			}
			if(dense_rowIdx23 > numNodes) {
				fragB0[0][2] = 0.0; fragB0[0][3] = 0.0;
				fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
				fragB0[2][2] = 0.0; fragB0[2][3] = 0.0;
				fragB0[3][2] = 0.0; fragB0[3][3] = 0.0;
			} else {
				vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
				vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
				vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
				vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
				fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
				fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
				fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
				fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
				vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
				vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
				vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
				vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
				fragB0[2][2] = load_fp32_from_global(d_MatB + sourceIdx4); 
				fragB0[2][3] = load_fp32_from_global(d_MatB + sourceIdx5);
				fragB0[3][2] = load_fp32_from_global(d_MatB + sourceIdx6); 
				fragB0[3][3] = load_fp32_from_global(d_MatB + sourceIdx7);
			}        
		// }
		tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
		tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
		tf32_m16n8k8(fragB0[2], fragA, fragC[2]);
		tf32_m16n8k8(fragB0[3], fragA, fragC[3]);
	} else {
		tf32_m16n8k8(fragB1[0], fragA, fragC[0]);
		tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
		tf32_m16n8k8(fragB1[2], fragA, fragC[2]);
		tf32_m16n8k8(fragB1[3], fragA, fragC[3]);
	}
	
	// if(local_warpID < dimTileNum) {
		// vint outOff = (bid << 3) * feature_dim + local_warpID * COL_WINDOW_R * 4 + offY;
		vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
		#pragma unroll
		for(vint i = 0; i < 4; ++i) {
			rowC = (tID_in_group << 1) + (i & 0x1);
			if(i < 2) colC = groupID;
			else colC = groupID + 8;
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 2, fragC[2][i]);
			atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 3, fragC[3][i]);
		}
	// }
}

/************************************************************************/
/* HOST FUNCTIONS */

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
) {
	vint threshold = 512;

	// printf("grid_size: %d, %d, %d\n", grid_size.x, grid_size.y, grid_size.z);
	// printf("block_size: %d, %d, %d\n", block_size.x, block_size.y, block_size.z);
	if(feature_dim <= threshold)  
		tf32_computeX128TransposePipe2<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	else
		tf32_computeX128TransposePipe2G128<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);

	// if(feature_dim <= threshold) {        
	//     // std::cout << " transpose pipeline..." << std::endl;
	//     // WARM UP TIME
	//     for(int i = 0; i < WARMUP_TIME; ++i) {
	//         tf32_computeX128TransposePipe2<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, 
	//         d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	//     }
	//     cudaDeviceSynchronize();
	//     // timer.Start();
	//     for(int i = 0; i < EXE_TIME; ++i) {
	//         // printf("exe time %d\n", i);
	//         tf32_computeX128TransposePipe2<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, 
	//         d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	//     }
	//     // timer.Stop();
	//     cudaDeviceSynchronize();
	// } else {
	//     // std::cout << " transpose pipeline22..." << std::endl;
	//     // WARM UP TIME
	//     for(int i = 0; i < WARMUP_TIME; ++i) {
	//         tf32_computeX128TransposePipe2G128<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, 
	//         d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	//     }
	//     cudaDeviceSynchronize();
	//     // timer.Start();
	//     for(int i = 0; i < EXE_TIME; ++i) {
	//         tf32_computeX128TransposePipe2G128<<<grid_size, block_size>>>(d_tcLocalBit, d_sparseA2B, d_dataA, 
	//         d_block2Idx, d_data2Idx, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	//     }
	//     // timer.Stop();
	//     cudaDeviceSynchronize();
	// }
}

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
) {
	vint threshold = 128;

	// std::cout << "doing AdpBalance pipelining ..." << std::endl;
	if(feature_dim <= threshold)
		tf32_computeX128TransposeAdpBalancePipe<<<grid_size, block_size>>>(d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
	else
		tf32_computeX128TransposeAdpBalancePipeG128<<<grid_size, block_size>>>(d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);

	// if(feature_dim <= threshold) {
	//     for(vint i = 0; i < EXE_TIME; ++i) {
	//         tf32_computeX128TransposeAdpBalancePipe<<<grid_size, block_size>>>(
	//             d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
	//             d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
	//             numNodes, feature_dim
	//         );
	//     }
	//     cudaDeviceSynchronize();
	//     // timer.Start();
	//     for(vint i = 0; i < EXE_TIME; ++i) {
	//         tf32_computeX128TransposeAdpBalancePipe<<<grid_size, block_size>>>(
	//             d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
	//             d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
	//             numNodes, feature_dim
	//         );
	//     }
	//     // timer.Stop();
	//     cudaDeviceSynchronize();
	// } else {
	//     for(vint i = 0; i < EXE_TIME; ++i) {
	//         tf32_computeX128TransposeAdpBalancePipeG128<<<grid_size, block_size>>>(
	//             d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
	//             d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
	//             numNodes, feature_dim
	//         );
	//     }
	//     cudaDeviceSynchronize();
	//     // timer.Start();
	//     for(vint i = 0; i < EXE_TIME; ++i) {
	//         tf32_computeX128TransposeAdpBalancePipeG128<<<grid_size, block_size>>>(
	//             d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
	//             d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
	//             numNodes, feature_dim
	//         );
	//     }
	//     // timer.Stop();
	//     cudaDeviceSynchronize();
	// }
}

/************************************************************************/
