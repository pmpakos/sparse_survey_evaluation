#ifndef GNNPILOT_V2_H
#define GNNPILOT_V2_H

#define SM_NUM 80
#define BLOCK_NUM 160
#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define BLOCK_SIZE_ALIGN 128

#define THREAD_NUM (BLOCK_NUM * BLOCK_SIZE)
#define WARP_NUM (THREAD_NUM / WARP_SIZE)
#define WARP_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define SHARED_EMBEDDING_SIZE 32

// Tile size for fused kernel
#define M_TILE_SIZE 96
#define N_TILE_SIZE 32
#define K_TILE_SIZE 32

#define N_BUF_TILE 32
#define K_BUF_TILE 32

#define WARP_ITER_SIZE 1
#define SINGLE_PCK_THRESH 256

#include <stdint.h>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>   // cudaEvent APIs

#define CUDA_CHECK_ERROR(call)\
{\
	cudaError_t _error = (cudaError_t)(call);\
	if(_error != cudaSuccess)\
	{\
		printf("*** CUDA Error *** at [%s:%d] error=%d, reason:%s \n",\
			__FILE__, __LINE__, _error, cudaGetErrorString(_error));\
	}\
}

using namespace std;

#define kg_max(a, b) (((a) >= (b))? (a): (b))
#define kg_min(a, b) (((a) < (b))? (a): (b))

typedef struct block_info_
{
	int row_st;
	int row_ed;

	block_info_() {}
	block_info_(int row_st_in, int row_ed_in): row_st(row_st_in), row_ed(row_ed_in) {}

} block_info;

typedef struct block_info2_
{
	int row_st;
	int row_ed;
	int col_st = -1;
	int col_ed = -1;

	block_info2_() {}
	__host__ __device__ block_info2_(int row_st_in, int row_ed_in): row_st(row_st_in), row_ed(row_ed_in) {}
} block_info2;

typedef struct warp_info_
{
	int row_st;
	int row_ed;
	int col_st;
	int col_ed;

	warp_info_() {}
	warp_info_(int row_st_in, int row_ed_in, int col_st_in, int col_ed_in):
	row_st(row_st_in), row_ed(row_ed_in), col_st(col_st_in), col_ed(col_ed_in) {}
} warp_info;

// Special format for KG-GNN
typedef struct bin_pack_
{
	// package part
	int *PckPtr;
	int *PckCont;
	int Pckn;

	// sparse part
	int *RowPtr_sp;
	int *ColIdx_sp;
	int spn;

	// on host
	std::vector<int> PckPtr_h;
	std::vector<int> PckCont_h;

	std::vector<int> RowPtr_sp_h;
	std::vector<int> ColIdx_sp_h;

	// Auxilary array for scheduling
	std::vector<int> BinPtr_h;
	std::vector<int> BinLoad;
	std::vector<int> PckLoad;

	bin_pack_() {}
	void bin_pack_cpy(int *PckPtr_in, int *PckCont_in, int *RowPtr_sp_in, int *ColIdx_sp_in)
	{
		PckPtr = PckPtr_in;
		PckCont = PckCont_in;
		//Pckn = Pckn_in;
		RowPtr_sp = RowPtr_sp_in;
		ColIdx_sp = ColIdx_sp_in;
	}
} bin_pack;

// info for bin_pack
typedef struct bin_pack_info_
{
	int bp_st;
	int bp_ed;

	bin_pack_info_() {}
	bin_pack_info_(int bp_st_in, int bp_ed_in): bp_st(bp_st_in), bp_ed(bp_ed_in) {}
} bin_pack_info;

typedef struct ana_info_
{
	warp_info *winfo = NULL;
	int winfo_n;

	warp_info **sinfo = NULL;
	int *sinfo_n;

	block_info *binfo = NULL;
	int binfo_n;

	bin_pack *bp = NULL;
	bin_pack_info **bpinfo = NULL;
	int *bpinfo_n;
	bin_pack_info *bpinfo2 = NULL;
	int bpinfo_n2;
	warp_info *spinfo = NULL;
	int spinfo_n;

	ana_info_(block_info *binfo_in, int binfo_n_in): binfo(binfo_in), binfo_n(binfo_n_in) {}
	ana_info_(warp_info **sinfo_in, int *sinfo_n_in): sinfo(sinfo_in), sinfo_n(sinfo_n_in) {}
	ana_info_(warp_info *winfo_in, int winfo_n_in): winfo(winfo_in), winfo_n(winfo_n_in) {}
	ana_info_(bin_pack *bp_in, bin_pack_info **bpinfo_in, int *bpinfo_n_in, warp_info *spinfo_in, int spinfo_n_in):
	bp(bp_in), bpinfo(bpinfo_in), bpinfo_n(bpinfo_n_in), spinfo(spinfo_in), spinfo_n(spinfo_n_in) {}
	ana_info_(bin_pack *bp_in, bin_pack_info *bpinfo2_in, int bpinfo_n2_in, warp_info *spinfo_in, int spinfo_n_in):
	bp(bp_in), bpinfo2(bpinfo2_in), bpinfo_n2(bpinfo_n2_in), spinfo(spinfo_in), spinfo_n(spinfo_n_in) {}
} ana_info;

/************************************************************************/
/* FUNCTION DECLARATIONS */

void kg_gcn_finalize(int64_t ana_add);
int64_t get_ana(int *row_ptr, int *col_idx, int m, int nnz, int feat_dim, int balance, std::string& op_name);
// void preprocess_gpu_wrapper();

/************************************************************************/

// aggregate.cu - naive
__global__ void gcn_aggregate_kernel_naive(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat);

// aggregate.cu - balanced
template <int FEAT_LEN>
__global__ void gcn_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st, int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n);

// aggregate_sddmm.cu
template <int FEAT_LEN>
__global__ void sddmm_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st, int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n);

/************************************************************************/

#endif // GNNPILOT_V2_H
