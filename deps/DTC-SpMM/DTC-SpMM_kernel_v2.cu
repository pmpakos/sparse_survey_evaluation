#include "DTC-SpMM_kernel_v2.h"

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
#include <vector>
#define WPB 8
#define EXE_TIME 128
// #define NUM_SM_GPU 128 // 4090
#define NUM_SM_GPU 108 // A100 (cslab)
using namespace nvcuda;

//////////////////////////////////////////////////////////////////////
// God knows why the fuck this is needed... I don't even want to bother explaining this shit.
namespace c10 {
	namespace detail {

		// Dummy for torchInternalAssertFail
		void __attribute__((weak)) torchInternalAssertFail(const char* expr, const char* file, unsigned int line, const char* function, const std::string& message)
		{
			printf("[Dummy torchInternalAssertFail] %s at %s:%u (%s): %s\n", expr, file, line, function, message.c_str());
			std::abort();  // <--- force it to not return
		}

		// Dummy for torchCheckFail
		void __attribute__((weak)) torchCheckFail(const char* expr, const char* file, unsigned int line, const std::string& message)
		{
			printf("[Dummy torchCheckFail] %s at %s:%u: %s\n", expr, file, line, message.c_str());
			std::abort();  // <--- force it to not return
		}

	} // namespace detail
} // namespace c10
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
__global__ void roundup_to_multiple_of_eight(int *input, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < size) {
		int rounded_value = ((input[tid] + 7) / 8) * 8;
		input[tid] = rounded_value;
	}
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid, int *padded_offset, uint8_t *padded_tileid, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < size) {
		int s = ori_offset[tid];
		int e = ori_offset[tid + 1];
		int s1 = padded_offset[tid];
		for (int i = 0; i < e - s; i++) {
			padded_tileid[s1 + i] = ori_tileid[s + i];
		}
	}
}

__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer, int num_nodes)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int nid = tid / 32;
	int laneid = tid % 32;
	// check a valid node range.
	if (nid < num_nodes) {
		#pragma unroll
		for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
				eid += 32) {
			edgeToRow[eid] = nid;
		}
	}
}
/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h, int blockSize_w, int num_nodes)
{
	int tid = threadIdx.x;
	int winId = blockIdx.x; // each warp one window
	unsigned block_start = nodePointer[winId * blockSize_h];
	unsigned block_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = block_end - block_start;
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
		seg_out[block_start + idx] = winId;
	}
}
void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h, int blockSize_w, int num_nodes)
{
	int block_size = 512;
	int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
	fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset, int *tcblock_rowid, int num_row_windows) 
{
	int tid = threadIdx.x;
	int winId = blockIdx.x; // each warp one window
	unsigned block_start = rowwindow_offset[winId];
	unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
	unsigned num_blocks = block_end - block_start;
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
		tcblock_rowid[block_start + idx] = winId;
	}
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid, int num_row_windows)
{
	int block_size = 512;
	int window_count = num_row_windows;
	generate_tcblock_rowid<<<window_count, block_size>>>(rowwindow_offset, tcblock_rowid, num_row_windows);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target)
{
	int left = 0;
	int right = size - 1;
	while (left <= right) {
		int mid = left + (right - left) / 2;
		if (arr[mid] == target) {
			while (mid > 0 && arr[mid - 1] == target) {
				mid--;
			}
			return mid;
		} else if (arr[mid] < target) {
			left = mid + 1;
		} else {
			right = mid - 1;
		}
	}
	return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length, int *loc)
{
	int cur = 1;
	while (cur < length) {
		if (array[cur] != array[cur - 1]) {
			(*loc)++;
			array[(*loc)] = array[cur];
		}
		cur++;
	}
}
__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist, int *edgelist_sort, int *edgetocol, int *blockpartition, int *blocknum, int blockSize_h, int blockSize_w, int num_nodes)
{
	int winId = blockIdx.x; // each warp one window
	unsigned block_start = nodePointer[winId * blockSize_h];
	unsigned block_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = block_end - block_start;
	if (num_window_edges == 0)
		return;
	const unsigned threadPerBlock = blockDim.x * blockDim.y;
	int *start = edgelist_sort + block_start;
	int size = 0;
	inplace_deduplication(start, num_window_edges, &size);
	int num = (size + blockSize_w) / blockSize_w;
	atomicAdd(blocknum, num);
	blockpartition[winId] = num;
	for (unsigned idx = block_start; idx < block_end; idx += 1) {
		int index = binarysearch(start, size + 1, edgelist[idx]);
		edgetocol[idx] = index;
	}
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist, int *edgelist_sort, int *edgetocol, int *blockpartition, int *blocknum, int blockSize_h, int blockSize_w, int num_nodes)
{
	int block_size = 1;
	int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
	int block_size1 = 128;
	int block_count1 = (window_count + 127) / 128;
	generate_edgetocolumn<<<window_count, block_size>>>(
			nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
			blockSize_h, blockSize_w, num_nodes);
	// generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
	// edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
	// blockSize_w, num_nodes);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob(int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow, int *edgeList, int *tcblock_offset, uint8_t *tcblocktile_id, int *sparseatob, int max_block, int num_nodes, int blockSize_h, int blockSize_w, int num_row_windows)
{
	extern __shared__ int pos_ptr[];
	int tid = threadIdx.x;
	int winId = blockIdx.x; // each warp one window
	unsigned block_start = rowwindow_offset[winId];
	unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
	unsigned num_blocks = block_end - block_start;
	if (num_blocks == 0) {
		return;
	}
	int *tcblock_offset_ptr = pos_ptr + num_blocks;
	int *tcblock_offset_global_ptr = tcblock_offset + block_start;
	int *tcblock_nnz_ptr = pos_ptr + num_blocks + 1;
	unsigned element_start = nodePointer[winId * blockSize_h];
	unsigned element_end =
		nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
	unsigned num_window_edges = element_end - element_start;
	if (num_window_edges == 0) {
		return;
	}
	for (int i = 0; i < 2 * num_blocks + 1; i++) {
		pos_ptr[i] = 0;
	}
	for (unsigned e_index = element_start; e_index < element_end; e_index++) {
		unsigned col = edgeToColumn[e_index]; // new col
		tcblock_nnz_ptr[col / blockSize_w]++;
	}
	for (int i = 0; i < num_blocks; i++) {
		tcblock_offset_global_ptr[i] = tcblock_nnz_ptr[i];
	}
	auto tileid = tcblocktile_id + element_start;
	auto sparse_AToB = sparseatob + block_start * blockSize_w;
	for (int i = 0; i < num_blocks; i++) {
		tcblock_nnz_ptr[i] += tcblock_nnz_ptr[i - 1];
	}
	for (unsigned e_index = element_start; e_index < element_end; e_index++) {
		unsigned col = edgeToColumn[e_index]; // new col
		unsigned tcblock_id = col / blockSize_w;
		unsigned row_local = edgeToRow[e_index] % blockSize_h;
		unsigned col_local = col % blockSize_w;
		tileid[tcblock_offset_ptr[tcblock_id] + pos_ptr[tcblock_id]] =
			(uint8_t)(row_local * blockSize_w + col_local);
		sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
		pos_ptr[tcblock_id]++;
	}
}
void generate_tcoffset_id_atob_cuda(int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow, int *edgeList, int *tcblock_offset, uint8_t *tcblock_tileid, int *sparseatob, int max_block, int num_nodes, int blockSize_h, int blockSize_w, int num_row_windows)
{
	int block_size = 1;
	int window_count = num_row_windows;
	const int dynamic_shared_size = (2 * max_block + 1) * sizeof(int);
	// std::cout << "dynamic_shared_size: " << dynamic_shared_size << std::endl;
	if (dynamic_shared_size > 98304) {
		int maxbytes = 131072; // 96 KB
		cudaFuncSetAttribute(generate_tcoffset_id_atob,
				cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	} else if (dynamic_shared_size > 65536) {
		int maxbytes = 98304; // 96 KB
		cudaFuncSetAttribute(generate_tcoffset_id_atob,
				cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	} else if (dynamic_shared_size > 32768) {
		int maxbytes = 65536; // 128 KB
		cudaFuncSetAttribute(generate_tcoffset_id_atob,
				cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
	}
	generate_tcoffset_id_atob<<<window_count, block_size, dynamic_shared_size>>>(
			nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
			tcblock_offset, tcblock_tileid, sparseatob, max_block, num_nodes,
			blockSize_h, blockSize_w, num_row_windows);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}
void padding_up_8(int *input, int size)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid, int *padded_offset, uint8_t *padded_tileid, int size)
{
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
			ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}
/*main function*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
seg_sort_dequ(int *seg, int *edgeLists, int *nodepointer, int *edgetocol, int *edgetorow, int *blockpartition, int *block_num, int *rowwindow_offset, int blockSize_h, int blockSize_w, int num_nodes, int num_edges, int rowwindow_num)
{
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(deviceSeg.end(), deviceEL.end()));
	thrust::sort(thrust::device, begin, end);
	generate_edgetocolumn_cuda(nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]), edgetocol, blockpartition, block_num, blockSize_h, blockSize_w, num_nodes);
	thrust::device_ptr<int> blockpartition_ptr = thrust::device_pointer_cast(blockpartition);
	thrust::device_ptr<int> rowwindow_offset_ptr = thrust::device_pointer_cast(rowwindow_offset + 1);
	thrust::device_vector<int> blockpartition_vector(blockpartition_ptr, blockpartition_ptr + rowwindow_num);
	thrust::inclusive_scan(blockpartition_vector.begin(), blockpartition_vector.end(), rowwindow_offset_ptr);
	auto options_gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_unit8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
	thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
	thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
	int block_counter = bnum_vector[0];
	auto tcblock_rowid_tensor = torch::zeros({block_counter}, options_gpu);
	auto tcblock_rowid = tcblock_rowid_tensor.data_ptr<int>(); // auto tcblock_rowid = tcblock_rowid_tensor.data<int>();
	generate_tcblock_rowid_cuda(rowwindow_offset, tcblock_rowid, rowwindow_num);
	auto max_element = thrust::max_element(thrust::device, blockpartition_vector.begin(), blockpartition_vector.end());
	int max_blocks = *max_element;
	auto tcblocktile_id_tensor = torch::zeros({num_edges}, options_gpu_unit8);
	auto tcblock_offset_tensor = torch::zeros({block_counter + 1}, options_gpu);
	auto sparse_AToX_index_tensor = torch::zeros({block_counter * blockSize_w}, options_gpu);
	auto tcblock_offset = tcblock_offset_tensor.data_ptr<int>(); // auto tcblock_offset = tcblock_offset_tensor.data<int>();
	auto sparse_AToX_index = sparse_AToX_index_tensor.data_ptr<int>(); // auto sparse_AToX_index = sparse_AToX_index_tensor.data<int>();
	auto tcblocktile_id = tcblocktile_id_tensor.data_ptr<uint8_t>(); // auto tcblocktile_id = tcblocktile_id_tensor.data<uint8_t>();
	generate_tcoffset_id_atob_cuda(nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists, tcblock_offset + 1, tcblocktile_id, sparse_AToX_index, max_blocks, num_nodes, blockSize_h, blockSize_w, rowwindow_num);
	thrust::device_ptr<int> tcblock_offset_ptr = thrust::device_pointer_cast(tcblock_offset);
	thrust::inclusive_scan(tcblock_offset_ptr, tcblock_offset_ptr + block_counter + 1, tcblock_offset_ptr);
	return std::make_tuple(tcblock_offset_tensor, tcblock_rowid_tensor, tcblocktile_id_tensor, sparse_AToX_index_tensor, block_counter);
}
void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes)
{
	int wrap_size = 32;
	int block_size = 1024;
	int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
	fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

/*
// From TC-GNN
int preprocess(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor, int num_nodes, int blockSize_h, int blockSize_w, torch::Tensor blockPartition_tensor, torch::Tensor edgeToColumn_tensor, torch::Tensor edgeToRow_tensor)
{
	// input tensors.
	auto edgeList = edgeList_tensor.accessor<int, 1>();
	auto nodePointer = nodePointer_tensor.accessor<int, 1>();

	// output tensors.
	auto blockPartition = blockPartition_tensor.accessor<int, 1>();
	auto edgeToColumn = edgeToColumn_tensor.accessor<int, 1>();
	auto edgeToRow = edgeToRow_tensor.accessor<int, 1>();
	auto start = std::chrono::high_resolution_clock::now();
	
	unsigned block_counter = 0;
	#pragma omp parallel for 
	for (unsigned nid = 0; nid < num_nodes; nid++){
		for (unsigned eid = nodePointer[nid]; eid < nodePointer[nid+1]; eid++)
			edgeToRow[eid] = nid;
	}
	#pragma omp parallel for reduction(+:block_counter)
	for (unsigned iter = 0; iter < num_nodes + 1; iter +=	blockSize_h){
		unsigned windowId = iter / blockSize_h;
		unsigned block_start = nodePointer[iter];
		unsigned block_end = nodePointer[min(iter + blockSize_h, num_nodes)];
		unsigned num_window_edges = block_end - block_start;
		unsigned *neighbor_window = (unsigned *) malloc (num_window_edges * sizeof(unsigned));
		memcpy(neighbor_window, &edgeList[block_start], num_window_edges * sizeof(unsigned));

		// Step-1: Sort the neighbor id array of a row window.
		thrust::sort(neighbor_window, neighbor_window + num_window_edges);

		// Step-2: Deduplication of the edge id array.
		// printf("Before dedupblication: %d\n", num_window_edges);
		std::map<unsigned, unsigned> clean_edges2col = inplace_deduplication(neighbor_window, num_window_edges);
		
		// generate blockPartition --> number of TC_blcok in each row window.
		blockPartition[windowId] = (clean_edges2col.size() + blockSize_w - 1) /blockSize_w;
		block_counter += blockPartition[windowId];

		// scan the array and generate edge to column mapping. --> edge_id to compressed_column_id of TC_block.
		for (unsigned e_index = block_start; e_index < block_end; e_index++){
			unsigned eid = edgeList[e_index];
			edgeToColumn[e_index] = clean_edges2col[eid];
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "\t CPU	original Preprocess time: " << elapsed_seconds.count() << " seconds\n";
	printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
	return block_counter;
}
*/

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
preprocess_gpu(torch::Tensor edgeList_tensor, torch::Tensor nodePointer_tensor, int num_nodes, int blockSize_h, int blockSize_w, torch::Tensor blockPartition_tensor, torch::Tensor edgeToColumn_tensor, torch::Tensor edgeToRow_tensor)
{
	// input tensors.
	auto options_gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_unit8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
	auto edgeList = edgeList_tensor.data_ptr<int>(); // auto edgeList = edgeList_tensor.data<int>();
	auto blockPartition = blockPartition_tensor.data_ptr<int>(); // auto blockPartition = blockPartition_tensor.data<int>();
	auto row_window_offset_tensor = torch::zeros({blockPartition_tensor.size(0) + 1}, options_gpu);
	auto row_window_offset = row_window_offset_tensor.data_ptr<int>(); // auto row_window_offset = row_window_offset_tensor.data<int>();
	auto edgeToColumn = edgeToColumn_tensor.data_ptr<int>(); // auto edgeToColumn = edgeToColumn_tensor.data<int>();
	auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, options_gpu);
	auto blocknum = torch::zeros({1}, options_gpu);
	auto block_num = blocknum.data_ptr<int>(); // auto block_num = blocknum.data<int>();
	auto edgeToRow = edgeToRow_tensor.data_ptr<int>(); // auto edgeToRow = edgeToRow_tensor.data<int>();
	auto nodePointer = nodePointer_tensor.data_ptr<int>(); // auto nodePointer = nodePointer_tensor.data<int>();
	auto seg_out = seg_out_tensor.data_ptr<int>(); // auto seg_out = seg_out_tensor.data<int>();
	auto start = std::chrono::high_resolution_clock::now();
	fill_edgeToRow_cuda(edgeToRow, nodePointer, num_nodes);
	int block_counter = 0;
	fill_segment_cuda(nodePointer, seg_out, blockSize_h, blockSize_w, num_nodes);
	auto tuple_tensor_blockcnt = seg_sort_dequ(
		seg_out, edgeList, nodePointer, edgeToColumn, edgeToRow, blockPartition,
		block_num, row_window_offset, blockSize_h, blockSize_w, num_nodes,
		edgeList_tensor.size(0), blockPartition_tensor.size(0));
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	// std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count() << " seconds\n";
	auto tcblock_offset_tensor = std::get<0>(tuple_tensor_blockcnt);
	auto tcblock_rowid_tensor = std::get<1>(tuple_tensor_blockcnt);
	auto tcblocktile_id_tensor = std::get<2>(tuple_tensor_blockcnt);
	auto sparse_AToX_index_tensor = std::get<3>(tuple_tensor_blockcnt);
	block_counter = std::get<4>(tuple_tensor_blockcnt);
	// printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter, block_counter * 8 * 16);
	return std::make_tuple(row_window_offset_tensor, tcblock_rowid_tensor, tcblocktile_id_tensor, tcblock_offset_tensor, sparse_AToX_index_tensor, block_counter);
}

//////////////////////////////////////////////////////////////////////

void check_cuda(torch::Tensor t, std::string name) {
	if (t.device().is_cuda()) {
		std::cout << "Tensor " << name << " is on CUDA (device) memory.\n";
	} else {
		std::cout << "Tensor " << name << " is on CPU (host) memory.\n";
	}
}

void print_first(torch::Tensor tensor, const char* name, int count)
{
	auto tensor_cpu = tensor.to(torch::kCPU);
	std::cout << "First ten elements of " << name << " (dtype = " << tensor.dtype() << " size = " << tensor.sizes() << "): [";

	torch::ScalarType dtype = tensor.scalar_type();
	switch (dtype) {
		case torch::kInt32: {
			auto data = tensor_cpu.data_ptr<int>();
			for (int i = 0; i < std::min<int64_t>(count, tensor_cpu.numel()); ++i)
				std::cout << data[i] << " ";
			break;
		}
		case torch::kInt64: {
			auto data = tensor_cpu.data_ptr<int64_t>();
			for (int i = 0; i < std::min<int64_t>(count, tensor_cpu.numel()); ++i)
				std::cout << data[i] << " ";
			break;
		}
		case torch::kFloat32: {
			auto data = tensor_cpu.data_ptr<float>();
			for (int i = 0; i < std::min<int64_t>(count, tensor_cpu.numel()); ++i)
				std::cout << data[i] << " ";
			break;
		}
		case torch::kFloat64: {
			auto data = tensor_cpu.data_ptr<double>();
			for (int i = 0; i < std::min<int64_t>(count, tensor_cpu.numel()); ++i)
				std::cout << data[i] << " ";
			break;
		}
		case torch::kUInt8: {
			auto data = tensor_cpu.data_ptr<uint8_t>();
			for (int i = 0; i < std::min<int64_t>(count, tensor_cpu.numel()); ++i)
				std::cout << static_cast<int>(data[i]) << " "; // print as int
			break;
		}
		default:
			std::cout << "Unsupported tensor dtype." << std::endl;
			return;
	}
	std::cout << "]\n";
}

template <typename T>
void copy_tensor_to_cpu(torch::Tensor tensor, T *output) {
	auto tensor_cpu = tensor.to(torch::kCPU);
	auto tensor_data = tensor_cpu.data_ptr<T>();
	for (int i = 0; i < tensor.size(0); i++) {
		output[i] = tensor_data[i];
	}
}

void preprocess_gpu_wrapper(int *row_ptr, int *col_idx,  int m, int n, int nnz, 
	int * num_row_windows_out, int * blockSize_h_out, int * blockSize_w_out, 
	int **RowWindowOffset_ptr_out, int **TCblockRowid_ptr_out, uint8_t **TCblocktileId_ptr_out, int **TCblockoffset_ptr_out, int **SparseAToXindex_ptr_out,
	int *RowWindowOffset_size_out, int *TCblockRowid_size_out, int *TCblocktileId_size_out, int *TCblockoffset_size_out, int *SparseAToXindex_size_out) 
{
	// Allocate memory for edgeList_tensor and nodePointer_tensor on the GPU
	auto edgeList_tensor = torch::from_blob(col_idx, {nnz}, torch::kInt32).to(torch::kCUDA);
	auto nodePointer_tensor = torch::from_blob(row_ptr, {m + 1}, torch::kInt32).to(torch::kCUDA);
	
	int num_nodes = m;
	int blockSize_h = BLK_H;
	int blockSize_w = BLK_W;

	int num_row_windows = (num_nodes + blockSize_h - 1) / blockSize_h;
	// Just initialize them to 0
	auto options_gpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto blockPartition_tensor = torch::zeros({num_row_windows}, options_gpu);
	auto edgeToColumn_tensor = torch::zeros({nnz}, options_gpu);
	auto edgeToRow_tensor = torch::zeros({nnz}, options_gpu);

	auto [RowWindowOffset, TCblockRowid, TCblocktileId, TCblockoffset, SparseAToXindex, block_count] = preprocess_gpu(edgeList_tensor, nodePointer_tensor, num_nodes, blockSize_h, blockSize_w, blockPartition_tensor, edgeToColumn_tensor, edgeToRow_tensor);

	*num_row_windows_out = num_row_windows;
	*blockSize_h_out = blockSize_h;
	*blockSize_w_out = blockSize_w;

	int * RowWindowOffset_ptr = (int *)malloc(RowWindowOffset.numel() * sizeof(int));
	copy_tensor_to_cpu<int>(RowWindowOffset, RowWindowOffset_ptr);

	int * TCblockRowid_ptr = (int *)malloc(TCblockRowid.numel() * sizeof(int));
	copy_tensor_to_cpu<int>(TCblockRowid, TCblockRowid_ptr);

	uint8_t * TCblocktileId_ptr = (uint8_t *)malloc(TCblocktileId.numel() * sizeof(uint8_t));
	copy_tensor_to_cpu<uint8_t>(TCblocktileId, TCblocktileId_ptr);

	int * TCblockoffset_ptr = (int *)malloc(TCblockoffset.numel() * sizeof(int));
	copy_tensor_to_cpu<int>(TCblockoffset, TCblockoffset_ptr);
	
	int * SparseAToXindex_ptr = (int *)malloc(SparseAToXindex.numel() * sizeof(int));
	copy_tensor_to_cpu<int>(SparseAToXindex, SparseAToXindex_ptr);

	*RowWindowOffset_ptr_out = RowWindowOffset_ptr;
	*TCblockRowid_ptr_out = TCblockRowid_ptr;
	*TCblocktileId_ptr_out = TCblocktileId_ptr;
	*TCblockoffset_ptr_out = TCblockoffset_ptr;
	*SparseAToXindex_ptr_out = SparseAToXindex_ptr;

	*RowWindowOffset_size_out = RowWindowOffset.numel();
	*TCblockRowid_size_out = TCblockRowid.numel();
	*TCblocktileId_size_out = TCblocktileId.numel();
	*TCblockoffset_size_out = TCblockoffset.numel();
	*SparseAToXindex_size_out = SparseAToXindex.numel();
}

//////////////////////////////////////////////////////////////////////

/************************************************************************/
/* FUNCTION DEFINITIONS */

// spmm_forward_ptx_uint8_improved --> spmm_forward_improved_ptx_uint8_cuda --> 
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		const float *__restrict__ input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		// if (tid < BLK_W) {
		//   sparse_AToX_index[tid] = numNodes + 1;
		// }
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;

		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(&valuesA[eIdx]));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off1 + off] = frag_D[i];
			output[o_off2 + off] = frag_D[i + 4];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 	// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,								// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off + off] = frag_D[i];
			output[o_off + off + 1] = frag_D[i + 4];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float2_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 6);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H + off_y;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}


		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off + off] = frag_D[i];
			output[o_off + off + 1] = frag_D[i + 4];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[4]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[5]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[2]), "r"(frag_B[6]), 
				"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[3]), "r"(frag_B[7]), 
				"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
				);
		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[4]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[5]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[2]), "r"(frag_B[6]), 
			"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[3]), "r"(frag_B[7]), 
			"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
			uint32_t off = row_d * embedding_dim + col_d;
			uint32_t off_set = o_off + off;
			output[off_set] = frag_D[i];
			output[off_set + 1] = frag_D[i + 4];
			output[off_set + 2] = frag_D[i + 8];
			output[off_set + 3] = frag_D[i + 12];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[4]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[5]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[2]), "r"(frag_B[6]), 
				"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[3]), "r"(frag_B[7]), 
				"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
				);
		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[4]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[5]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[2]), "r"(frag_B[6]), 
			"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[3]), "r"(frag_B[7]), 
			"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
			uint32_t off = row_d * embedding_dim + col_d;
			uint32_t off_set = o_off + off;
			output[off_set] = frag_D[i];
			output[off_set + 1] = frag_D[i + 4];
			output[off_set + 2] = frag_D[i + 8];
			output[off_set + 3] = frag_D[i + 12];
		}
}

// spmm_balance_forward_ptx_uint8_prefetch --> spmm_balance_forward_cuda_ptx_unit8_prefetch
__global__ void spmm_forward_cuda_kernel_improved_ptx_uint8_v1_strict_balance_withv(
		const int *__restrict__ TCblock_rowid, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA,
		const int tc_count,
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		const float *__restrict__ input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned wid_BLK_H = wid * BLK_H;
	unsigned off = wid_BLK_H * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid_BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
		int ptr = lb + idx;
		if (ptr < hb) {
			tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
		}
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	for (unsigned j = lb; j < hb; j++) {
		current_rid = tc_rowid[j - lb];
		unsigned eIdx_start = TCblock_offset[j];			
		unsigned eIdx_end = TCblock_offset[j + 1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	 
		if (current_rid != former_row_id) {
			uint32_t o_off1 = former_row_id * BLK_H * embedding_dim + wid_BLK_H;
			uint32_t o_off2 = o_off1 + 8;
			if (wid < dimTileNum)
				#pragma unroll
				for(int i = 0; i < 4; i++) {
					uint32_t row_d = 0;
					if( i < 2 ) {
						row_d = group_id;
					} else {
						row_d = group_id + 8;
					}
					uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
					uint32_t off = row_d * embedding_dim + col_d;
					atomicAdd(output + o_off1 + off, frag_D[i]);
					atomicAdd(output + o_off2 + off, frag_D[i + 4]);
					frag_D[i] = 0.0;
					frag_D[i + 4] = 0.0;
				}
			former_row_id = current_rid;
		}
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = (numNodes + 1) * embedding_dim;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(int)TCblocktile_id[eIdx]] = valuesA[eIdx];		// set the edge of the sparse_A.	
		}
		#pragma unroll
		for (unsigned eIdx = sparse_AToX_idx_start + tid; eIdx < sparse_AToX_idx_start + BLK_W; eIdx += threadPerBlock) {
			sparse_AToX_index[tid] = sparse_AToX_idx[eIdx] * embedding_dim;	
		}
		__syncthreads();
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[sparse_A_idx3]));
		__syncthreads();
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[dense_rowIdx_off];						// TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));
			asm volatile(
					"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
					: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
					: "r"(frag_A[0]), "r"(frag_A[1]), 
					"r"(frag_B[0]), 
					"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
					);
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));
			asm volatile(
					"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
					: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
					: "r"(frag_A[0]), "r"(frag_A[1]),
					"r"(frag_B[1]), 
					"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
					);

			dense_rowIdx = sparse_AToX_index[dense_rowIdx_off1];						// TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));
			asm volatile(
					"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
					: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
					: "r"(frag_A[2]), "r"(frag_A[3]), 
					"r"(frag_B[2]), 
					"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
					);
			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
			asm volatile(
					"mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
					: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
					: "r"(frag_A[2]), "r"(frag_A[3]), 
					"r"(frag_B[3]), 
					"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
					);
		}
	}
	uint32_t o_off1 = current_rid * BLK_H * embedding_dim + wid_BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			atomicAdd(output + o_off1 + off, frag_D[i]);
			atomicAdd(output + o_off2 + off, frag_D[i + 4]);
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_with_value_double_buffer_float4_split_balance(
		const int *__restrict__ TCblock_rowid, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const float *__restrict__ valuesA, 		
		const int tc_count,
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = bid * TCBLOCK_PER_WARP;
	const unsigned hb = min((bid + 1) * TCBLOCK_PER_WARP, tc_count);
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	__shared__ int tc_rowid[TCBLOCK_PER_WARP];
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	#pragma unroll
	for (unsigned idx = tid; idx < TCBLOCK_PER_WARP; idx += threadPerBlock) {
		int ptr = lb + idx;
		if (ptr < hb) {
			tc_rowid[idx] = __ldg(TCblock_rowid + ptr);
		}
	}
	__syncthreads();
	unsigned former_row_id = tc_rowid[0];
	unsigned current_rid = former_row_id;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = valuesA[eIdx];  // set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim; // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim; // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			int id_local = (((int)TCblocktile_id[eIdx])<<2);
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa_ptr + id_local + (smem_sel_next << 9)), "l"(valuesA+eIdx));	  
		}
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}

		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[4]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[5]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[2]), "r"(frag_B[6]), 
				"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[3]), "r"(frag_B[7]), 
				"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
				);


		current_rid = tc_rowid[j - lb];
		if (current_rid != former_row_id) {
			uint32_t o_off = former_row_id * BLK_H * embedding_dim + wid * 32 + off_y;
			if (wid < dimTileNum)
				#pragma unroll
				for(int i = 0; i < 4; i++) {
					uint32_t row_d = 0;
					if( i < 2 ) {
						row_d = group_id;
					} else {
						row_d = group_id + 8;
					}
					uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
					uint32_t off = row_d * embedding_dim + col_d;
					uint32_t off_set = o_off + off;
					atomicAdd(output + off_set, frag_D[i]);
					atomicAdd(output + off_set + 1, frag_D[i + 4]);
					atomicAdd(output + off_set + 2, frag_D[i + 8]);
					atomicAdd(output + off_set + 3, frag_D[i + 12]);
					frag_D[i] = 0.0;
					frag_D[i + 4] = 0.0;
					frag_D[i + 8] = 0.0;
					frag_D[i + 12] = 0.0;
				}
			former_row_id = current_rid;
		}

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}
	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[4]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[5]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[2]), "r"(frag_B[6]), 
			"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[3]), "r"(frag_B[7]), 
			"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
			);

	uint32_t o_off = current_rid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
			uint32_t off = row_d * embedding_dim + col_d;
			uint32_t off_set = o_off + off;
			atomicAdd(output + off_set, frag_D[i]);
			atomicAdd(output + off_set + 1, frag_D[i + 4]);
			atomicAdd(output + off_set + 2, frag_D[i + 8]);
			atomicAdd(output + off_set + 3, frag_D[i + 12]);
		}
}

// spmm_forward_ptx_uint8_improved_for_gcn --> spmm_forward_improved_ptx_uint8_cuda_dtc_for_gcn
__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float4_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.	
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 7);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / 32;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 4 + wid * 32 + off_y;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[8]; // 8 * 8 * 2  / 32 = 4
	float frag_D[16] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j - lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
			} else {
				float4 t = FLOAT4(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
			}

		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   

		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx+sparse_AToX_idx_start+tid));	
		}
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[4]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[5]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[2]), "r"(frag_B[6]), 
				"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[3]), "r"(frag_B[7]), 
				"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
				);
		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.w));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(z));
		} else {
			float4 t = FLOAT4(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[4]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[5]) : "f"(t.y));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[6]) : "f"(t.z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[7]) : "f"(t.w));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[4]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[5]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[8]), "=f"(frag_D[9]), "=f"(frag_D[10]), "=f"(frag_D[11])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[2]), "r"(frag_B[6]), 
			"f"(frag_D[8]), "f"(frag_D[9]), "f"(frag_D[10]), "f"(frag_D[11])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[12]), "=f"(frag_D[13]), "=f"(frag_D[14]), "=f"(frag_D[15])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[3]), "r"(frag_B[7]), 
			"f"(frag_D[12]), "f"(frag_D[13]), "f"(frag_D[14]), "f"(frag_D[15])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * 32 + off_y;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group << 3) + ((i & 0x1)<<2);
			uint32_t off = row_d * embedding_dim + col_d;
			uint32_t off_set = o_off + off;
			output[off_set] = frag_D[i];
			output[off_set + 1] = frag_D[i + 4];
			output[off_set + 2] = frag_D[i + 8];
			output[off_set + 3] = frag_D[i + 12];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2_split(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	int off_y = (blockIdx.y << 6);
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H + off_y;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;  // TC_block_col to dense_tile_row.
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;  // TC_block_col to dense_tile_row.
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H + off_y;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off + off] = frag_D[i];
			output[o_off + off + 1] = frag_D[i + 4];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer_float2(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.		
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		float *input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned off = wid * BLK_W * BLK_H;
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) * 2 + wid * BLK_H;
	// unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	  
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
			}
			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;

			if (source_idx >= dense_bound) {
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			} else {
				float2 t = FLOAT2(input[source_idx]);
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
			}
		}

		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(sparse_AToX_idx + sparse_AToX_idx_start + tid));	
		}
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(t.y));
		}
		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound) {
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		} else {
			float2 t = FLOAT2(input[source_idx]);
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(t.x));
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(t.y));
		}
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);

	uint32_t o_off = bid * BLK_H * embedding_dim + wid * BLK_H;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group<<2) + ((i & 0x1)<<1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off + off] = frag_D[i];
			output[o_off + off + 1] = frag_D[i + 4];
		}
}

__global__ void spmm_forward_cuda_kernel_improved_ptx_1684_uint8_v1_double_buffer(
		const int *__restrict__ Rowwindow_offset, 		// offset of each row window.
		const uint8_t *__restrict__ TCblocktile_id, 		// id of each TC block nonzero element.
		const int *__restrict__ TCblock_offset, 		// colid of each TC block nonzero element.
		const int *__restrict__ sparse_AToX_idx, 		// colid of each TC block nonzero element.
		const int numNodes,
		const int numEdges,
		const int embedding_dim,					// embedding dimension.
		const float *__restrict__ input,			// input feature matrix.
		float *output								// output feature matrix.
		)
{
	int bid = blockIdx.x;								// block_index == row_window_index
	const unsigned lb = Rowwindow_offset[bid];
	const unsigned hb = Rowwindow_offset[bid + 1];
	if(lb == hb) return;
	const unsigned wid = threadIdx.y;								// warp_index handling multi-dimension > 16. (local warp id)
	const unsigned laneid = threadIdx.x;							// lanid of each warp.
	const unsigned tid = threadIdx.y * blockDim.x + laneid;			// threadid of each block. (local thread idx)
	const unsigned warpSize = blockDim.x;							// number of threads per warp.
	const unsigned threadPerBlock = blockDim.x * blockDim.y;		// number of threads per block.
	const unsigned dimTileNum = embedding_dim / BLK_H;					// number of tiles along the dimension
	const unsigned dense_bound = numNodes * embedding_dim;
	__shared__ float sparse_A[2*BLK_H * BLK_W];					// row-major sparse matrix shared memory store.
	__shared__ int sparse_AToX_index[2*BLK_W];					// TC_block col to dense_tile row.
	unsigned dense_rowIdx_off = (laneid % 4);
	unsigned dense_rowIdx_off1 = dense_rowIdx_off + 4;
	unsigned dense_dimIdx = (laneid / 4) + wid * BLK_H;
	unsigned dense_dimIdx1 = dense_dimIdx + 8;
	uint32_t group_id = (laneid >> 2);
	uint32_t tid_in_group = (laneid % 4);
	uint32_t sparse_A_idx = (group_id << 3) + tid_in_group;
	uint32_t sparse_A_idx1 = ((group_id+8) << 3) + tid_in_group;
	uint32_t sparse_A_idx2 = (group_id << 3) + tid_in_group + 4;
	uint32_t sparse_A_idx3 = ((group_id+8) << 3) + tid_in_group + 4;
	int shuffle_idx = (laneid/4) + (laneid%4)*8;
	uint32_t frag_A[4]; // 16 * 8  / 32 = 4
	uint32_t frag_B[4]; // 8 * 8 * 2  / 32 = 4
	float frag_D[8] = {0.0}; // 16 * 16 / 32 = 8
	float z = 0.0;
	int sa_ptr = __cvta_generic_to_shared(sparse_A);
	int si_ptr = __cvta_generic_to_shared(sparse_AToX_index);
	unsigned eIdx_start = TCblock_offset[lb];			
	unsigned eIdx_end = TCblock_offset[lb+1];
	// pre loop
	{
		unsigned sparse_AToX_idx_start = lb * BLK_W;	 
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[idx] = 0.0;
		}
		__syncthreads();
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(int)TCblocktile_id[eIdx]] = 1.0;		// set the edge of the sparse_A.	  
		}
		if (tid < BLK_W) {
			sparse_AToX_index[tid] = sparse_AToX_idx[sparse_AToX_idx_start + tid];	
		}
		__syncthreads();
	}
	//main loop
	for (unsigned j = lb + 1; j < hb; j++) {
		int smem_sel = ((j - lb) & 1) ^ 1;
		int smem_sel_next = ( (j-lb - 1) & 1) ^ 1;
		if (wid < dimTileNum) {
			unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
			unsigned source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

			dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
			source_idx = dense_rowIdx + dense_dimIdx;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

			source_idx = dense_rowIdx + dense_dimIdx1;
			if (source_idx >= dense_bound)
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
			else
				asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
		}
		eIdx_start = TCblock_offset[j];			
		eIdx_end = TCblock_offset[j+1];
		unsigned sparse_AToX_idx_start = j * BLK_W;	   
		#pragma unroll
		for (unsigned idx = tid; idx < BLK_W * BLK_H; idx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + idx] = 0.0;
		}
		__syncthreads();
		if (tid < BLK_W) {	
			asm ("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(si_ptr + (tid<<2) + (smem_sel_next << 5)), "l"(&sparse_AToX_idx[sparse_AToX_idx_start + tid]));	
		}
		#pragma unroll
		for (unsigned eIdx = eIdx_start + tid; eIdx < eIdx_end; eIdx += threadPerBlock) {
			sparse_A[(smem_sel_next << 7) + TCblocktile_id[eIdx]] = 1.0;	  
		}
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
		asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));

		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[0]), "r"(frag_B[2]), 
				"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
				);
		asm volatile(
				"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
				: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
				: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
				"r"(frag_B[1]), "r"(frag_B[3]), 
				"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
				);

		asm ("cp.async.commit_group;\n"::);
		asm ("cp.async.wait_group 0;\n" ::);
		__syncthreads();
	}

	//end loop
	int smem_sel = ((hb - lb) & 1) ^ 1;
	if (wid < dimTileNum) {
		unsigned dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off] * embedding_dim;
		unsigned source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[0]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[1]) : "f"(input[source_idx]));

		dense_rowIdx = sparse_AToX_index[(smem_sel <<3) + dense_rowIdx_off1] * embedding_dim;
		source_idx = dense_rowIdx + dense_dimIdx;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[2]) : "f"(input[source_idx]));

		source_idx = dense_rowIdx + dense_dimIdx1;
		if (source_idx >= dense_bound)
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(z));
		else
			asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_B[3]) : "f"(input[source_idx]));
	}
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[0]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[1]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx1]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[2]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx2]));
	asm("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(frag_A[3]) : "f"(sparse_A[(smem_sel << 7) + sparse_A_idx3]));
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[0]), "r"(frag_B[2]), 
			"f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3])
			);
	asm volatile(
			"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5, %6, %7}, {%8,%9}, {%10,%11,%12,%13};\n"
			: "=f"(frag_D[4]), "=f"(frag_D[5]), "=f"(frag_D[6]), "=f"(frag_D[7])
			: "r"(frag_A[0]), "r"(frag_A[1]), "r"(frag_A[2]), "r"(frag_A[3]), 
			"r"(frag_B[1]), "r"(frag_B[3]), 
			"f"(frag_D[4]), "f"(frag_D[5]), "f"(frag_D[6]), "f"(frag_D[7])
			);


	uint32_t o_off1 = bid * BLK_H * embedding_dim + wid * BLK_H;
	uint32_t o_off2 = o_off1 + 8;
	if (wid < dimTileNum)
		#pragma unroll
		for(int i = 0; i < 4; i++) {
			uint32_t row_d = 0;
			if( i < 2 ) {
				row_d = group_id;
			} else {
				row_d = group_id + 8;
			}
			uint32_t col_d = (tid_in_group * 2) + (i & 0x1);
			uint32_t off = row_d * embedding_dim + col_d;
			output[o_off1 + off] = frag_D[i];
			output[o_off2 + off] = frag_D[i + 4];
		}
}

/************************************************************************/
