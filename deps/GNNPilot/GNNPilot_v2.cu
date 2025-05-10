#include "GNNPilot_v2.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>
#include <torch/torch.h>
using namespace nvcuda;
// using namespace std;
#include <cmath>  // for ceil()

//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////

/********************************************************************/
/* preprocessing.cu */
/********************************************************************/
// Only partite the neighbours of high-degree nodes
void kg_csr_balance(int m, int nnz, int *RowPtr, int wsize, warp_info **winfo, int *winfo_n)
{
	std::vector<warp_info> host_winfo;
	for (int row = 0; row < m; row++)
	{
		int row_st = RowPtr[row];
		int row_ed = RowPtr[row + 1];
		// printf(">>> row = %d (%d - %d)\n", row, row_st, row_ed);

		int wi;
		for (wi = row_st; wi < row_ed - wsize; wi += wsize)
		{
			warp_info tmp = warp_info(row, row + 1, wi, wi + wsize);
			host_winfo.push_back(tmp);
		}
		warp_info tmp = warp_info(row, row + 1, wi, row_ed);
		host_winfo.push_back(tmp);
	}
	//printf("warp num %d size %d\n", host_winfo.size(), host_winfo.size() * sizeof(warp_info));

	*winfo_n = host_winfo.size();
	cudaMalloc(winfo, host_winfo.size() * sizeof(warp_info));
	cudaMemcpy(*winfo, &host_winfo[0], host_winfo.size() * sizeof(warp_info), cudaMemcpyHostToDevice);
}

// Partite nodes into equally sized groups
void kg_csr_balance2(int m, int nnz, int *RowPtr, int wsize, int alpha, warp_info **winfo, int *winfo_n)
{
	std::vector<warp_info> host_winfo;
	std::vector<int> warp_load;

	int group_n = 0;
	int last_start_row = 0;
	int last_end_row = -1;
	int last_start_col, last_end_col;

	for (int row = 0; row < m; row++)
	{
		int row_st = RowPtr[row];
		int row_ed = RowPtr[row + 1];

		// if (row < 50) printf("%d %d %d %d\n", row_st, row_ed, wsize, group_n);

		// Approximation of a write back
		// int alpha = 15;

		if (row_ed - row_st + alpha > wsize - group_n || last_end_row == -1)
		{
			//printf("?\n");
			//printf("%d\n", last_end_row);
			
			if (last_end_row != -1)
			{
				if (wsize - group_n <= alpha)
				//if (true)
				{
					warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
					host_winfo.push_back(tmp);
					warp_load.push_back(group_n);
					group_n = 0;
				}
				else
				{
					last_end_col += wsize - group_n - alpha;
					row_st += wsize - group_n - alpha;
					last_end_row = row + 1;
					group_n += wsize - group_n;
					warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
					host_winfo.push_back(tmp);
					warp_load.push_back(group_n);
					group_n = 0;
				}
			}

			// if (host_winfo.size() < 50)
			//     printf("%d row %d row_ed %d col %d col_ed %d\n", host_winfo.size(), 
			//     last_start_row, last_end_row, last_start_col, last_end_col);

			int wi;
			for (wi = row_st; wi < row_ed - wsize + alpha; wi += wsize - alpha)
			{
				warp_info tmp = warp_info(row, row + 1, wi, wi + wsize - alpha);
				host_winfo.push_back(tmp);
				warp_load.push_back(wsize);
				group_n = 0;
			}
			last_start_row = row;
			last_start_col = wi;
			group_n += row_ed - wi + alpha;
		}
		else
			group_n += row_ed - row_st + alpha;

		last_end_row = row + 1;
		last_end_col = row_ed;
	}
	// if (last_start_row < last_end_row && last_end_row < m)
	// {
	warp_info tmp = warp_info(last_start_row, last_end_row, last_start_col, last_end_col);
	host_winfo.push_back(tmp);
	warp_load.push_back(group_n);
	//}

	// for (int i = 0; i < host_winfo.size(); i++)
	// {
	//     if (host_winfo.size() < 50)
	//         printf("%d row %d row_ed %d col %d col_ed %d\n", i,
	//         host_winfo[i].row_st, host_winfo[i].row_ed, host_winfo[i].col_st, host_winfo[i].col_ed);
	// }

	// printf("total count %d\n", host_winfo.size());

	*winfo_n = host_winfo.size();
	cudaMalloc(winfo, host_winfo.size() * sizeof(warp_info));
	cudaMemcpy(*winfo, &host_winfo[0], host_winfo.size() * sizeof(warp_info), cudaMemcpyHostToDevice);

	int warp_min_load = -1;
	int warp_max_load = 0;

	for (int i = 0; i < warp_load.size(); i++)
	{
		int tmp = warp_load[i];

		if (warp_min_load == -1 || tmp < warp_min_load) warp_min_load = tmp;
		if (warp_max_load < tmp) warp_max_load = tmp;

		//printf("%d ", tmp);
	}

	//printf("warp load %d %d\n", warp_min_load, warp_max_load);
}

void sinfo2device(std::vector<warp_info> *host_sinfo, warp_info ***sinfo, int **sinfo_n)
{
	int host_sinfo_n[WARP_NUM];
	for (int i = 0; i < WARP_NUM; i++)
		host_sinfo_n[i] = host_sinfo[i].size();
	
	cudaMalloc(sinfo_n, WARP_NUM * sizeof(int));
	cudaMemcpy(*sinfo_n, host_sinfo_n, WARP_NUM * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(sinfo, WARP_NUM * sizeof(warp_info*));

	warp_info *info_h[WARP_NUM];
	for (int i = 0; i < WARP_NUM; i++)
	{
		int level = host_sinfo_n[i];
		if (level)
		{
			// printf("host wid %d level %d\n", i, level);
			// printf("host info st %d ed %d\n", host_sinfo[i][0].row_st, host_sinfo[i][0].row_ed);
			cudaMalloc(&info_h[i], level * sizeof(warp_info));
			cudaMemcpy(info_h[i], &host_sinfo[i][0], level * sizeof(warp_info), cudaMemcpyHostToDevice);
		}
		else
			info_h[i] = NULL;
	}

	cudaMemcpy(*sinfo, info_h, WARP_NUM * sizeof(warp_info*), cudaMemcpyHostToDevice);
}

void kg_finalize_cu(ana_info* ana)
{

	if (ana->winfo)
	{
		CUDA_CHECK_ERROR(cudaFree(ana->winfo));
	}
	

	if (ana->sinfo)
	{
		warp_info* host_sinfo_n[WARP_NUM];
		CUDA_CHECK_ERROR(cudaMemcpy(host_sinfo_n, ana->sinfo_n, WARP_NUM * sizeof(warp_info*), cudaMemcpyDeviceToHost));

		for (int i = 0; i < WARP_NUM; i++)
			CUDA_CHECK_ERROR(cudaFree(host_sinfo_n[i]));

		CUDA_CHECK_ERROR(cudaFree(ana->sinfo));
		CUDA_CHECK_ERROR(cudaFree(ana->sinfo_n));
	}

	if (ana->binfo)
	{
		CUDA_CHECK_ERROR(cudaFree(ana->binfo));
	}

	if (ana->bp)
	{
		CUDA_CHECK_ERROR(cudaFree(ana->bp->PckPtr));
		CUDA_CHECK_ERROR(cudaFree(ana->bp->PckCont));
		CUDA_CHECK_ERROR(cudaFree(ana->bp->RowPtr_sp));
		CUDA_CHECK_ERROR(cudaFree(ana->bp->ColIdx_sp));
		
		if (ana->bpinfo)
		{
			bin_pack_info *host_sinfo_n[WARP_NUM];
			CUDA_CHECK_ERROR(cudaMemcpy(host_sinfo_n, ana->bpinfo, WARP_NUM * sizeof(bin_pack_info*), cudaMemcpyDeviceToHost));

			for (int i = 0; i < WARP_NUM; i++)
			{
				//if (i >= 10) continue;
				if (host_sinfo_n[i])
					CUDA_CHECK_ERROR(cudaFree(host_sinfo_n[i]));
			}

			CUDA_CHECK_ERROR(cudaFree(ana->bpinfo));
			CUDA_CHECK_ERROR(cudaFree(ana->bpinfo_n));
		}

		if (ana->bpinfo2)
		{
			CUDA_CHECK_ERROR(cudaFree(ana->bpinfo2));
		}

		if (ana->spinfo)
			CUDA_CHECK_ERROR(cudaFree(ana->spinfo));
	
		delete ana->bp;
	}
}

/********************************************************************/
/* gnn_analysis.cpp */
/********************************************************************/
int64_t kg_gcn_balance(torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t wsize)
{
	int m = RowPtr.size(0) - 1;
	int nnz = ColIdx.size(0);
	int winfo_n;
	warp_info *winfo;
	kg_csr_balance(m, nnz, RowPtr.data_ptr<int>(), wsize, &winfo, &winfo_n);
	ana_info *ret = new ana_info(winfo, winfo_n);
	return (int64_t)ret;
}

int64_t kg_gcn_balance2(torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t wsize, int64_t alpha)
{
	int m = RowPtr.size(0) - 1;
	int nnz = ColIdx.size(0);
	int winfo_n;
	warp_info *winfo;
	if (wsize < alpha)
	{
		printf("Wrong parameters (wsize < alpha). Set wsize = alpha + 1.\n");
		wsize = alpha + 1;
	}
	kg_csr_balance2(m, nnz, RowPtr.data_ptr<int>(), wsize, alpha, &winfo, &winfo_n);
	ana_info *ret = new ana_info(winfo, winfo_n);
	return (int64_t)ret;
}

void kg_gcn_finalize(int64_t ana_add)
{
	ana_info* ptr = (ana_info*)ana_add;

	kg_finalize_cu(ptr);

	delete ptr;
}
//////////////////////////////////////////////////////////////////////

int64_t get_ana(int *row_ptr, int *col_idx, int m, int nnz, int feat_dim, int balance, std::string& op_name)
{
	// Allocate memory for edgeList_tensor and nodePointer_tensor. This time on the CPU!
	auto row_ptr_tensor = torch::from_blob(row_ptr, {m + 1}, torch::kInt32);
	auto col_idx_tensor = torch::from_blob(col_idx, {nnz}, torch::kInt32);

	double avg_rnz = nnz*1.0/m;

	if(balance == -1){
		if(avg_rnz > 100)
			balance = 6;
		else
			balance = 2;
	}

	int64_t ana_info;

	if(balance){
		if(balance == 1)
			ana_info = kg_gcn_balance(row_ptr_tensor, col_idx_tensor, 32);
		else if(balance == 2)
		{
			int64_t alpha = 0;
			double delta = 0;
			double np = 0;

			if(op_name == "SpMM"){
				alpha = 10;
				delta = 4;
				if(avg_rnz < 16)
					np = 128;
				else if (avg_rnz < 128)
					np = 256;
				else
					np = 512;
				np = np / std::ceil(1.0 * feat_dim / 32 / 2);
			}
			else if(op_name == "SDDMM"){
				alpha = 0;
				delta = 2;
				np = std::pow(2, static_cast<int>(std::log2(avg_rnz + 1)) + delta);
			}
			if(np < 32)
				np = 32;
			ana_info = kg_gcn_balance2(row_ptr_tensor, col_idx_tensor, (int64_t) np, alpha);
		}
		/*
		else if(balance == 3)
			ana_info = kg_gcn_balance3(row_ptr_tensor, col_idx_tensor, 128, 15);
		else if(balance == 4)
			ana_info = kg_gcn_balance4(row_ptr_tensor, col_idx_tensor, 1024);
		else if(balance == 5)
			ana_info = kg_gcn_schedule_locality(row_ptr_tensor, col_idx_tensor, 1024);
		else if(balance == 6)
			ana_info = kg_gcn_bin_pack(row_ptr_tensor, col_idx_tensor, 1024, 256, 20, 32, 64, 10);
		*/
	}
	return ana_info;
}

//////////////////////////////////////////////////////////////////////

/************************************************************************/
/* FUNCTION DEFINITIONS */

// aggregate.cu - naive
__global__ void gcn_aggregate_kernel_naive(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat)
{
	int local_tid = threadIdx.x;
	int global_tid = blockIdx.x * blockDim.x + local_tid;
	//int local_wid = local_tid / WARP_SIZE;
	int global_wid = global_tid / WARP_SIZE;
	int lane_id = local_tid & (WARP_SIZE - 1);

	int row = global_wid;

	//if (!lane_id) printf("warp_id %d row %d\n", global_wid, row);

	int start_ptr = RowPtr[row];
	int end_ptr = RowPtr[row + 1];
	//if (start_ptr + 32 < end_ptr) end_ptr = start_ptr + 32;
	// float degree_inv = 1.0 ;/// (end_ptr - start_ptr);

	if (row < m)
	{
		int self_idx = row * feat_len;
		for (int j = lane_id; j < feat_len; j += WARP_SIZE)
		//for (int j = lane_id; j < WARP_SIZE; j+=WARP_SIZE)
		{
			float result = 0.0;
			for (int i = start_ptr; i < end_ptr; i++)
			{
				int nid = ColIdx[i];
				int feat_idx = nid * feat_len + j;
				float degree_inv = Values[i];
				result += in_feat[feat_idx] * degree_inv;
			}
			out_feat[self_idx + j] = result;
		}
	}
}

// aggregate.cu - balanced
template <int FEAT_LEN>
__global__ void gcn_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st, int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
	int local_tid = threadIdx.x;
	if (local_tid >= BLOCK_SIZE) return;
	//int global_tid = blockIdx.x * blockDim.x + local_tid;
	int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
	//int local_wid = local_tid / WARP_SIZE;
	int global_wid = global_tid / WARP_SIZE;
	int lane_id = local_tid & (WARP_SIZE - 1);

	//if (lane_id >= feat_size) return;

	int j_st = feat_st + FEAT_LEN * blockIdx.y + lane_id;
	if (j_st >= feat_len) return;
	int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

	for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
	{
		if (tgt >= winfo_n) return;
		warp_info info = winfo[tgt];
		for (int row = info.row_st; row < info.row_ed; row++)
		{
			int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
			int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;
			// float degree_inv = 1.0 ;
			int self_idx = row * feat_len;
			for (int j = j_st; j < j_ed; j += WARP_SIZE)
			{
				float result = 0.0;
				for (int i = start_ptr; i < end_ptr; i++)
				{
					int nid = ColIdx[i];
					int feat_idx = nid * feat_len + j;
					float degree_inv = Values[i];
					result += in_feat[feat_idx] * degree_inv;
				}
				atomicAdd(&out_feat[self_idx + j], result);
			}
			//__syncthreads();
		}
	}
}

void gcn_aggregate_balance(int m, int nnz, int feat_len, int *RowPtr, int *ColIdx, float *Values, float *in_feat, float *out_feat, warp_info* winfo, int winfo_n)
{
	// neighbour grouping for balance
	int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
	int thread_num = warp_num * WARP_SIZE;
	int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

	const int kernel_len = 32;
	dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
	dim3 block(BLOCK_SIZE_ALIGN);
	int feat_st = 0;
	gcn_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st, RowPtr, ColIdx, Values, in_feat, out_feat, winfo, winfo_n);
}

// aggregate_sddmm.cu
template <int FEAT_LEN>
__global__ void sddmm_aggregate_kernel_balance_aligned(int m, int nnz, int feat_len, int feat_st, int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n)
{
	int local_tid = threadIdx.x;
	if (local_tid >= BLOCK_SIZE) return;
	int local_wid = local_tid / WARP_SIZE;
	int global_tid = blockIdx.x * BLOCK_SIZE + local_tid;
	int global_wid = global_tid / WARP_SIZE;
	int lane_id = local_tid & (WARP_SIZE - 1);

	int j_st = feat_st + FEAT_LEN * blockIdx.y;
	if (j_st + lane_id >= feat_len) return;
	int j_ed = kg_min(j_st + FEAT_LEN, feat_len);

	// printf("active %d\n", lane_id);

	unsigned mask = __activemask();

	__shared__ float buffer_all[FEAT_LEN * WARP_PER_BLOCK];
	float *m1_buffer = buffer_all + FEAT_LEN * local_wid;

	for (int tgt = global_wid * WARP_ITER_SIZE; tgt < (global_wid + 1) * WARP_ITER_SIZE; tgt++)
	{
		if (tgt >= winfo_n) return;
		warp_info info = winfo[tgt];

		// if (!lane_id) printf("wid %d st %d ed %d\n", global_wid, info.row_st, info.row_ed);

		for (int row = info.row_st; row < info.row_ed; row++)
		{
			int start_ptr = (RowPtr[row] > info.col_st)? RowPtr[row]: info.col_st;
			int end_ptr = (RowPtr[row + 1] < info.col_ed)? RowPtr[row + 1]: info.col_ed;

			for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
			{
				m1_buffer[j - j_st] = in_feat1[row * feat_len + j];
			}

			for (int i = start_ptr; i < end_ptr; i++)
			{
				float result = 0.0;
				int nid = ColIdx[i];
				for (int j = j_st + lane_id; j < j_ed; j += WARP_SIZE)
				{
					result += m1_buffer[j - j_st] * in_feat2[nid * feat_len + j];
					// printf("tid %d row %d m1_buffer[0] = %.2f in_feat2 = %.2f\n", lane_id, row, m1_buffer[j - j_st], in_feat2[nid * feat_len + j]);
				}
				// if (result) printf("tid %d result %.2f\n", lane_id, result);
				for (int k = 16; k > 0; k >>= 1)
				{
					result += __shfl_down_sync(mask, result, k);
				}
				if (lane_id == 0)
				{
					//out_feat[i] = result;
					atomicAdd(&out_feat[i], result);
				}
			}
		}
	}
}

void sddmm_aggregate_balance(int m, int nnz, int feat_len,
int *RowPtr, int *ColIdx, float *in_feat1, float *in_feat2, float *out_feat, warp_info* winfo, int winfo_n)
{
	// neighbour grouping for balance
	int warp_num = (winfo_n + WARP_ITER_SIZE - 1) / WARP_ITER_SIZE;
	int thread_num = warp_num * WARP_SIZE;
	int block_num = (thread_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

	const int kernel_len = 32;
	dim3 grid(block_num, (feat_len + kernel_len - 1) / kernel_len);
	dim3 block(BLOCK_SIZE_ALIGN);
	int feat_st = 0;
	sddmm_aggregate_kernel_balance_aligned<kernel_len><<<grid, block>>>(m, nnz, feat_len, feat_st, RowPtr, ColIdx, in_feat1, in_feat2, out_feat, winfo, winfo_n);
}

/************************************************************************/
