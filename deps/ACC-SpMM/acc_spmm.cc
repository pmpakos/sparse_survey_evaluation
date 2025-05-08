#include "acc_spmm_v2.h"
#include "acc_spmm.h"

// __host__
// DEPRECATED!!!
void tf32_spmm(
	METCFBit<MAT_VAL_TYPE>& metcf_bit, 
	BME<MAT_VAL_TYPE>& bme, 
	AdpBME<MAT_VAL_TYPE>& adpbme,
	COO<MAT_VAL_TYPE>* coo,
	const vint feature_dim, 
	const std::string filename, 
	bool load_balance = false
) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	// int num_sms = deviceProp.multiProcessorCount;
	// int num_tb_per_sm = deviceProp.maxBlocksPerMultiProcessor;

	std::cout << "Number of Streaming Multiprocessors (SM): " << deviceProp.multiProcessorCount << std::endl;
	std::cout << "Maximum number of blocks per SM: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;

	// vint tmp = metcf_bit.sparseA2B.size() % COL_WINDOW;
	std::cout << "metcf_bit.sparseA2B.size(): " << metcf_bit.sparseA2B.size() << std::endl;

	vint            rowWndSize, tcLocalBitSize, dataSize, sparseA2BSize;
	rowWndSize      =   static_cast<vint>(metcf_bit.rowWindowOffsetBit.size());
	tcLocalBitSize  =   static_cast<vint>(metcf_bit.tcLocalBit.size());
	dataSize        =   static_cast<vint>(metcf_bit.data.size());
	sparseA2BSize   =   static_cast<vint>(metcf_bit.sparseA2B.size());

	vint            groupOffsetSize, tcOffsetSize, rowIndicesSize;
	groupOffsetSize   =   static_cast<vint>(bme.groupOffset.size());
	tcOffsetSize      =   static_cast<vint>(bme.tcOffset.size());
	rowIndicesSize    =   static_cast<vint>(bme.rowIndices.size());

	TCLOCAL_TYPE* ptr_tcLocalBit        =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
	MAT_VAL_TYPE* ptr_data              =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
	vint* ptr_sparseA2B                 =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
	
	std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
	std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);
	std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);

	vint* ptr_group_offset       = (vint*)malloc(sizeof(vint) * groupOffsetSize);
	vint* ptr_tc_offset          = (vint*)malloc(sizeof(vint) * tcOffsetSize);
	vint* ptr_row_indices_offset = (vint*)malloc(sizeof(vint) * rowIndicesSize);
	
	std::copy(bme.groupOffset.begin(), bme.groupOffset.end(), ptr_group_offset);
	std::copy(bme.tcOffset.begin(), bme.tcOffset.end(), ptr_tc_offset);
	std::copy(bme.rowIndices.begin(), bme.rowIndices.end(), ptr_row_indices_offset);

	vint* ptr_adp_groupOffset   = (vint*)malloc(sizeof(vint) * static_cast<vint>(adpbme.groupOffset.size()));
	vint* ptr_adp_rowIndices    = (vint*)malloc(sizeof(vint) * static_cast<vint>(adpbme.rowIndices.size()));

	std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_adp_groupOffset);
	std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_adp_rowIndices);

	/*------------------------------------------------------*/
	vint numBlocks = 0;
	for(vint i = 0; i < rowWndSize; ++i) {
		if(metcf_bit.rowWindowOffsetBit[i] == false) ++numBlocks;
	}
	// init denseB
	vint numNodes = static_cast<vint>(coo->rows);
	// vint numEdges = static_cast<vint>(coo->nnz);

	vint denseC_size = std::max(numBlocks * ROW_WINDOW * feature_dim, (adpbme.rowIndices.back() + 8) * feature_dim);

	MAT_VAL_TYPE* DenseMatB = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * coo->cols * feature_dim);
	MAT_VAL_TYPE* DenseMatC = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);
	init_vec1(coo->cols * feature_dim, DenseMatB, 1.0);
	init_vec1(numBlocks * ROW_WINDOW * feature_dim, DenseMatC, 0.0);

	vint* block2Idx = (vint*)malloc(sizeof(vint) * (numBlocks + 1));
	memset(block2Idx, 0, sizeof(vint) * (numBlocks + 1));
	vint b_idx = 0;
	vint cnt_TCblock = 0;
	vint AvgTCBlock = rowWndSize / numBlocks;
	vint sumWindow = 0;
	vint tmp_zero = 0;
	vint tmp_nonZero = 0;
	for(vint i = 0; i < rowWndSize; ++i) {
		++cnt_TCblock;
		if(metcf_bit.rowWindowOffsetBit[i] == false) {
			block2Idx[++b_idx] = (i + 1);
			if(cnt_TCblock - AvgTCBlock == 0) {
				++tmp_zero;
			} else {
				++tmp_nonZero;
			}
			sumWindow += std::abs(int(cnt_TCblock - AvgTCBlock));
			cnt_TCblock = 0;
		}
	}
	vint IBD = sumWindow / numBlocks;

	if(IBD >= 8) {
		load_balance = true;
	} else {
		load_balance = false;
	}

	std::cout << "load balance is : " << load_balance << " " << IBD << std::endl;

	if(b_idx != numBlocks) {
		std::cerr << "Error with handling rowWndOffsetBit" << std::endl;
		return;
	}

	vint* block2IdxBME = (vint*)malloc(sizeof(vint) * (numBlocks + 1));
	memset(block2IdxBME, 0, sizeof(vint) * (numBlocks + 1));
	vint b_idxBME = 0;
	for(vint i = 0; i < rowWndSize; ++i) {
		if(metcf_bit.rowWindowOffsetBit[i] == false) {
			block2IdxBME[++b_idxBME] = (i + 1);
		}
	}
	if(b_idxBME != numBlocks) {
		std::cout << "Error with handling rowWndOffsetBit" << std::endl;
		return;
	}
	std::cout << " numBlocks is : " << numBlocks << std::endl;
	/*------------------------------------------------------*/

	vint* data2Idx = (vint*)malloc(sizeof(vint) * (rowWndSize + 1));
	memset(data2Idx, 0, sizeof(vint) * (rowWndSize + 1));
	// vint data_bidx = 1;
	vint cnt_data_idx = 0; 
	vint d_i = 0;
	printf("metcf_bit.tcLocalBit.size() = %ld, rowWndSize = %d\n", metcf_bit.tcLocalBit.size(), rowWndSize);
	for (vint i = 0; i < metcf_bit.tcLocalBit.size(); ++i) {
		if(d_i < rowWndSize){
			data2Idx[d_i++] = cnt_data_idx;
			cnt_data_idx += __builtin_popcountll(metcf_bit.tcLocalBit[i]);
			// printf("%d: cnt_data_idx = %d (metcf_bit.tcLocalBit = %ld) \n", i, cnt_data_idx, metcf_bit.tcLocalBit[i]);
			// std::cout << i << " cnt_data_idx = " << cnt_data_idx << "(metcf_bit.tcLocalBit = " << metcf_bit.tcLocalBit[i] <<  " -> " << __builtin_popcountll(metcf_bit.tcLocalBit[i]) << ")" << std::endl;
		}
		else{
			printf("d_i > rowWndSize\n");
			break;
		}
	}
	data2Idx[d_i] = cnt_data_idx;
	// std::cout << "metcf_bit.tcLocalBit.size() is : " << metcf_bit.tcLocalBit.size() << std::endl;

	/*------------------------------------------------------*/
	vint*   d_block2Idx;
	vint*   d_data2Idx;

	// BIT_TYPE*       d_rowWndOffsetBit;
	TCLOCAL_TYPE*   d_tcLocalBit;
	MAT_VAL_TYPE*   d_dataA;
	MAT_PTR_TYPE*   d_sparseA2B;
	// cudaMallocB
	MAT_VAL_TYPE*   d_DenseMatB;
	MAT_VAL_TYPE*   d_DenseMatC;
	MAT_PTR_TYPE*   d_group_offset, *d_tc_offset, *d_row_indices;
	MAT_PTR_TYPE*   d_adp_group_offset, *d_adp_row_indices;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data2Idx, sizeof(vint) * (rowWndSize + 1)));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_block2Idx, sizeof(vint) * (numBlocks + 1)));

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dataA, sizeof(MAT_VAL_TYPE) * dataSize));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(MAT_PTR_TYPE) * sparseA2BSize));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * coo->cols * feature_dim));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_group_offset, sizeof(MAT_PTR_TYPE) * groupOffsetSize));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tc_offset, sizeof(MAT_PTR_TYPE) * tcOffsetSize));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_indices, sizeof(MAT_PTR_TYPE) * rowIndicesSize));

	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_adp_group_offset, sizeof(MAT_PTR_TYPE) * adpbme.groupOffset.size()));
	CHECK_CUDA_ERROR(cudaMalloc((void**)&d_adp_row_indices, sizeof(MAT_PTR_TYPE) * adpbme.rowIndices.size()));

	CHECK_CUDA_ERROR(cudaMemcpy(d_block2Idx, block2Idx, sizeof(vint) * (numBlocks + 1), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_data2Idx, data2Idx, sizeof(vint) * (rowWndSize + 1), cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_dataA, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(MAT_PTR_TYPE) * sparseA2BSize, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * coo->cols * feature_dim, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_group_offset, ptr_group_offset, sizeof(MAT_PTR_TYPE) * groupOffsetSize, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_tc_offset, ptr_tc_offset, sizeof(MAT_PTR_TYPE) * tcOffsetSize, cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_row_indices, ptr_row_indices_offset, sizeof(MAT_PTR_TYPE) * rowIndicesSize, cudaMemcpyHostToDevice));

	CHECK_CUDA_ERROR(cudaMemcpy(d_adp_group_offset, ptr_adp_groupOffset, sizeof(MAT_PTR_TYPE) * adpbme.groupOffset.size(), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_adp_row_indices, ptr_adp_rowIndices, sizeof(MAT_PTR_TYPE) * adpbme.rowIndices.size(), cudaMemcpyHostToDevice));
	std::cout << vint(denseC_size) << std::endl;

	dim3 grid_size;
	dim3 block_size;

	if(!load_balance) {
		int warpsPerBlk = 0;
		vint threshold = 512;
		if(feature_dim <= threshold)
			warpsPerBlk = feature_dim / (COL_WINDOW_R << 1);
		else
			warpsPerBlk = feature_dim / (COL_WINDOW_R << 2);

		// dim3 grid_size(numBlocks, 1, 1);
		dim3 grid_size;
		grid_size.x = numBlocks;
		grid_size.y = 1;
		grid_size.z = 1;
		// dim3 block_size(WARP_SIZE, warpsPerBlk, 1);
		dim3 block_size;
		block_size.x = WARP_SIZE;
		block_size.y = warpsPerBlk;
		block_size.z = 1;
	} else {
		int warpsPerBlk = 0;
		vint threshold = 128;
		if(feature_dim <= threshold)
			warpsPerBlk = feature_dim / (COL_WINDOW_R << 1);
		else
			warpsPerBlk = feature_dim / (COL_WINDOW_R << 2);

		// dim3 grid_size(adpbme.groupOffset.size()-1, 1, 1);
		grid_size.x = adpbme.groupOffset.size()-1;
		grid_size.y = 1;
		grid_size.z = 1;
		// dim3 block_size(WARP_SIZE, warpsPerBlk, 1);
		block_size.x = WARP_SIZE;
		block_size.y = warpsPerBlk;
		block_size.z = 1;
	}

	GpuTimer timer2;
	timer2.Start();
	if(!load_balance) {
		printf("NO load_balance\n");
		tf32TransposeCompute(
			d_tcLocalBit, d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC, 
			d_block2Idx, d_data2Idx, 
			numNodes, numBlocks, feature_dim, grid_size, block_size);
	} else {
		printf("YES load_balance\n");
		tf32TransposeAdpBalanceCompute(
			d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
			d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC, 
			adpbme, numBlocks, numNodes, 
			feature_dim, denseC_size, grid_size, block_size);
	}
	timer2.Stop();

	float elapsed_time2 = 0.0;
	elapsed_time2 = timer2.Elapsed() / EXE_TIME;
	printf("Elapsed time: %8.4lf ms\n", elapsed_time2);

	CHECK_LAST_CUDA_ERROR();
	// gettimeofday(&t2, NULL);
	cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);
	// print_denseC(DenseMatC, coo->rows, numBlocks * ROW_WINDOW, feature_dim);
	float spmm_flop = float(coo->nnz) * float(feature_dim) * 2.0;
	float throughput_ = (float(spmm_flop * 1000.00)) / (elapsed_time2 * 1000. * 1000. * 1000.);

	std::ofstream outFile("./result/4090/AccSpMM-general1.csv", std::ios::app);

	if (!outFile) {
		std::cerr << "Error Opening AccSpMM-general1.csv" << std::endl;
	}
	outFile << filename << "," << feature_dim << "," << elapsed_time2 << "," << throughput_ << "\n";
	outFile.close();
	
	// free(ptr_row_wnd_offset_bit); 
	free(ptr_tcLocalBit); 
	free(ptr_sparseA2B); 
	free(ptr_data);
	free(DenseMatB);
	free(block2Idx);
	free(data2Idx);

	cudaFree(d_data2Idx); cudaFree(d_block2Idx);
	cudaFree(d_group_offset); cudaFree(d_tc_offset); cudaFree(d_row_indices);

	// cudaFree(d_rowWndOffsetBit); 
	cudaFree(d_tcLocalBit); cudaFree(d_dataA); cudaFree(d_sparseA2B);
	cudaFree(d_DenseMatB); cudaFree(d_DenseMatC);
}

// // __host__
// int main(int argc, char** argv) {
//     if(argc < 2) {
//         printf("Run the code by './mma_tf32 matrix.mtx.'\n");
//         return 0;
//     }
//     char* filename;
//     filename = argv[1];

//     // handle filename
//     std::string mtx_name = match_filename(std::string(filename));
//     std::string log_name = "./log/" + mtx_name + ".log";
//     std::string tmp_name = mtx_name + ".txt";
//     std::string reorderd_name = "./reordered_data/good_to_run/" + mtx_name + ".my_reorder.mtx";
//     // const char* reordered_file_name = reorderd_name.c_str();
//     std::ofstream logfile(log_name, std::ios::app);
//     printf("\n===%s===\n\n", filename);

//     // load original mtx
//     COO<MAT_VAL_TYPE>* coo = (COO<MAT_VAL_TYPE>*)malloc(sizeof(COO<MAT_VAL_TYPE>));
//     read_from_mtx<MAT_VAL_TYPE>(filename, coo);
//     // coo->show();
//     std::cout << coo->sort_matrix() << std::endl;

//     /*===================================================*/
//     const CSR<MAT_VAL_TYPE> csr = COO2CSR<MAT_VAL_TYPE>(coo);
	
//     METCF<MAT_VAL_TYPE> metcf;
//     metcf.CSR2METCF(csr);
//     METCFBit<MAT_VAL_TYPE> metcf_bit;
//     metcf_bit.METCF2METCFBit(metcf);
//     metcf_bit.printSize();
//     printf("tc num of METCF: %ld\n", metcf_bit.rowWindowOffsetBit.size());

//     BME<MAT_VAL_TYPE> bme;
//     bme.CSR2BME(csr);
//     // bme.show();
//     printf("tc num of BME: %ld\n", bme.tcLocalBit.size() * 64 / ROW_WINDOW / COL_WINDOW);

//     AdpBME<MAT_VAL_TYPE> adpbme;
//     adpbme.CSR2AdpBME(csr, 3, 1, 2);
//     /*===================================================*/

//     /*================TENSOR CORE================*/
//     // for(vint feature_dim = 128; feature_dim <= 512; feature_dim *= 2){
//     //     printf("feature_dim = %d\n", feature_dim);

//     //     tf32_spmm(metcf_bit, bme, adpbme, coo, feature_dim, mtx_name, false);
//     // }
//     printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
//     vint feature_dim = atoi(argv[2]);
//     tf32_spmm(metcf_bit, bme, adpbme, coo, feature_dim, mtx_name, false);
//     printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");
//     // cudaDeviceReset();

//     // load banlance
//     free(coo);
//     return 0;
// }
