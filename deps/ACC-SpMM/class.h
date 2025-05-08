#ifndef __CLASS_H__
#define __CLASS_H__

#include "common.h"

template <class dataType>
struct COO {
	vint        rows,   cols,   nnz;
	vint*       row,    *col;
	dataType*   data;
	
	void show() {
		for(vint i = 0; i < nnz; ++i) {
			printf("row = %d, col = %d, data = %lf\n", row[i], col[i], data[i]);
		}
	}

	bool is_sorted_matrix() {
		for(vint i = 0; i < nnz-1; ++i) {
			if((row[i] > row[i+1]) || (row[i] == row[i+1] && col[i] >= col[i+1])) {
				printf("not sorted: row: %u, col: %u", row[i], col[i]);
				printf("not sorted: row: %u, col: %u", row[i+1], col[i+1]);
				return false;
			}
		}
		return true;
	}

	bool sort_matrix() {
		std::vector<std::tuple<vint, vint, dataType>> tuples(nnz);
		for (vint i = 0; i < nnz; ++i) {
			tuples[i] = std::make_tuple(row[i], col[i], data[i]);
		}

		std::sort(tuples.begin(), tuples.end());

		for (vint i = 0; i < nnz; ++i) {
			row[i] = std::get<0>(tuples[i]);
			col[i] = std::get<1>(tuples[i]);
			data[i] = std::get<2>(tuples[i]);
		}

		return is_sorted_matrix();
	}

	// 写入文件 （行、列、权重）
	void write2file(const char* filename) {
		FILE* file = fopen(filename, "w");
		// char* outfile_name;

		if(file) {
			for(vint i = 0; i < nnz; ++i) {
				// fprintf(file, "%d %d %lf\n", row[i], col[i], data[i]);
				// if(row[i] < col[i]) continue;
				fprintf(file, "%u %u\n", row[i], col[i]);
			}
		} else {
			perror("Error opening file.");
		}
		fclose(file);
	}

	std::map<vint, std::vector<vint>> store_mtx_map() {
		std::cout << "is sorted mtx: " << sort_matrix()<< std::endl;
		std::map<vint, std::vector<vint>> mtx_map;
		vint r = 0;
		vint cnt = 0;
		for(vint i = 0; i < nnz && r <= rows;) {
			bool flag = false;
			while (row[i] == r && i < nnz) {
				mtx_map[r].push_back(static_cast<vint>(col[i]));
				++cnt; ++i;
				flag = true;
			}
			if(!flag) {
				mtx_map[r] = {};
			}
			++r;
		}
		return mtx_map;
	}

	vint cnt_mxn(vint m = ROW_WINDOW, vint n = COL_WINDOW, vint base_nnz = BASE_NNZ) {   // 调用 store_mtx_map() 之后再调用该函数
		// TODO count nums of m x n parts
		std::map<vint, std::vector<vint>> mp = store_mtx_map();
		vint tc_blocks = 0; vint greater16nnz = 0;
		std::vector<vint> cnt_nnz(cols, 0);
		for(vint row_idx = 0; row_idx < rows; row_idx += m) {
			vint start_row = row_idx, end_row = (start_row + m) > rows ? rows : (start_row + m);
			for(vint p_row = start_row; p_row < end_row; ++p_row) {
				std::vector<vint> vec_cols = mp[p_row];
				for(vint col_idx: vec_cols) {
					++cnt_nnz[col_idx];
				}
			}
			vint cnt_cols = 0; vint cnt_blk_nnz = 0;
			for(vint i = 0; i < cols; ++i) {
				if(cnt_nnz[i] > 0) {
					++cnt_cols;
					cnt_blk_nnz += cnt_nnz[i];
				}
				if(cnt_cols == n) {
					++tc_blocks;
					cnt_cols = 0;
					if(cnt_blk_nnz >= base_nnz) {
						++greater16nnz;
					}
					cnt_blk_nnz = 0;
				}
			}
			if(cnt_cols > 0) {
				++tc_blocks;
				if(cnt_blk_nnz >= base_nnz) {
					++greater16nnz;
				}
			}
			std::fill(cnt_nnz.begin(), cnt_nnz.end(), 0);
		}
		std::cout << "num of tc_blocks: " << tc_blocks << std::endl;
		std::cout << ">16 nnz: " << greater16nnz << " (" << std::fixed << std::setprecision(2) << double(static_cast<double>(greater16nnz) / tc_blocks) * 100.0 << "%) " <<  std::endl;
		return tc_blocks;
	}
	
	void remap_vertex(const char* filename, std::vector<vint>& ori2new) {
		FILE* file = fopen(filename, "w");
		if(file) {
			fprintf(file, "%%%%MatrixMarket matrix coordinate real general\n");
			fprintf(file, "%%-----------------------------------------------\n");
			fprintf(file, "%d %d %d\n", rows, cols, nnz);
			for(vint i = 0; i < nnz; ++i) {
				row[i] = ori2new[row[i]];
				col[i] = ori2new[col[i]];
				fprintf(file, "%u %u %lg\n", (row[i]+1), (col[i]+1), data[i]);
			}
			fclose(file);
		} else {
			perror("Error opening file in remap_vertex.\n");
		}
		// std::ofstream outFile;
		// outFile.open("myreordered_mtx.txt");
		// for(vint i = 0; i < nnz; ++i) {
		//     outFile << row[i] << " " << col[i] << std::endl;
		// }
		// outFile.close();
	}

	void remap_rabbit_vertex(std::unique_ptr<vint[]>& ori2new) {
		vint* ptr = ori2new.get();
		for(vint i = 0; i < static_cast<vint>(nnz); ++i) {
			row[i] = ptr[row[i]];
			col[i] = ptr[col[i]];
		}
	}

	~COO() {
		free(row); free(col); free(data);
	}
};

template <class dataType>
struct CSR {
	vint rows, cols, nnz;
	std::vector<vint> row_ptr;
	std::vector<vint> col_idx;
	std::vector<dataType> data;

	void show() {
		printf("Matrix in csr format, Row: %d, Col: %d, NNZ: %d\n", rows, cols, nnz);
		for(vint i = 0; i < row_ptr.size()-1; ++i) {
			printf("\nrow = %d\n", i);
			
			for(vint j = row_ptr[i]; j< row_ptr[i+1]; j++){
				printf("col = %d: data = %lf\n", col_idx[j], data[j]);
			}
		}
	}
	~CSR() {
		row_ptr.clear(); row_ptr.shrink_to_fit();
		col_idx.clear(); col_idx.shrink_to_fit();
		data.clear(); data.shrink_to_fit();
	}
};

/*=============================================*/
template <class dataType>
CSR<dataType> COO2CSR(COO<dataType>* coo) {
	CSR<dataType> csr;
	csr.rows = coo->rows;
	csr.cols = coo->cols;
	csr.nnz = coo->nnz;
	csr.row_ptr.resize(coo->rows + 1, 0); 
	csr.col_idx.resize(coo->nnz);
	csr.data.resize(coo->nnz);

	for (vint i = 0; i < coo->nnz; i++) {
		csr.row_ptr[coo->row[i]+1]++;
	}
	for (vint i = 0; i < coo->rows; i++) {
		csr.row_ptr[i+1] += csr.row_ptr[i]; 
	}

	for (vint i = 0; i < coo->nnz; i++) {
		vint row = coo->row[i];
		vint dest = csr.row_ptr[row];
		csr.data[dest] = coo->data[i];
		csr.col_idx[dest] = coo->col[i];
		csr.row_ptr[row]++;
	}

	// Restore row_ptr
	for (vint i = coo->rows; i > 0; i--) {
		csr.row_ptr[i] = csr.row_ptr[i - 1];
	}
	csr.row_ptr[0] = 0;

	return csr;
}

template <class dataType>
struct TCF {
	std::vector<vint>       rowWindowOffset;
	std::vector<vint>       tcOffset;
	std::vector<uint8_t>    tcLocalId;
	std::vector<vint>       sparseA2B;
	std::vector<dataType>   data; // 顺序跟sparseAtoB一样
};


template <class dataType>
struct METCF {
	std::vector<vint>       rowWindowOffset;
	std::vector<vint>       tcOffset;
	std::vector<uint8_t>    tcLocalId;
	std::vector<vint>       sparseA2B;
	std::vector<dataType>   data; // 顺序跟sparseAtoB一样

	void padListLength(std::vector<vint>& lst) {
		vint current_length = lst.size();
		vint remainder = current_length % COL_WINDOW;
		if (remainder != 0) {
			vint padding_size = COL_WINDOW - remainder;
			lst.insert(lst.end(), padding_size, UINT32_MAX);
		}
	}

	void CSR2METCF(const CSR<dataType>& csr) {

		vint num_nodes = csr.row_ptr.size() - 1;
		rowWindowOffset.push_back(0);

		for (vint iter = 0; iter < num_nodes; iter += ROW_WINDOW) {
			vint windowId = iter / ROW_WINDOW;
			vint block_start = csr.row_ptr[iter];
			vint block_end = csr.row_ptr[std::min(iter + ROW_WINDOW, num_nodes)];

			std::vector<vint> neighbor_window(csr.col_idx.begin() + block_start, csr.col_idx.begin() + block_end);
			std::sort(neighbor_window.begin(), neighbor_window.end());
			std::vector<vint> unique_edges;
			std::unique_copy(neighbor_window.begin(), neighbor_window.end(), std::back_inserter(unique_edges));

			std::unordered_map<vint, vint> clean_edges2col;
			for (vint i = 0; i < unique_edges.size(); ++i) {
				clean_edges2col[unique_edges[i]] = i;
			}

			padListLength(unique_edges);
			sparseA2B.insert(sparseA2B.end(), unique_edges.begin(), unique_edges.end());

			vint window_tc_num = (unique_edges.size() + COL_WINDOW - 1) / COL_WINDOW;
			rowWindowOffset.push_back(rowWindowOffset[windowId] + window_tc_num);

			tcOffset.resize(tcOffset.size() + window_tc_num, 0);
			
			std::vector<std::vector<uint8_t>> tcLocalIdtmp(window_tc_num);
			std::vector<std::vector<dataType>> datatmp(window_tc_num);
			for (vint r = iter; r < std::min(iter + ROW_WINDOW, num_nodes); ++r) {
				for (vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id) {
					vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
					vint offset_index = rowWindowOffset[windowId] + c_idx / COL_WINDOW;
					tcOffset[offset_index]++;
					tcLocalIdtmp[c_idx / COL_WINDOW].push_back((r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW);
					datatmp[c_idx / COL_WINDOW].push_back(csr.data[nnz_id]);
				}
			}

			for (vint i = 0; i < tcLocalIdtmp.size(); ++i) {
				tcLocalId.insert(tcLocalId.end(), tcLocalIdtmp[i].begin(), tcLocalIdtmp[i].end());
				data.insert(data.end(), datatmp[i].begin(), datatmp[i].end());
			}
		}

		tcOffset.insert(tcOffset.begin(), 0);
		std::partial_sum(tcOffset.begin(), tcOffset.end(), tcOffset.begin());
	}

	// void readMETCF(std::vector<std::vector<vint>>& tcCols, std::vector<std::vector<dataType>>& tcValues) {
	void printTCblock(bool printMsg = false) {

		std::vector<std::vector<vint>> tcCols(rowWindowOffset.back(), std::vector<vint>(ROW_WINDOW * COL_WINDOW, 0));
		std::vector<std::vector<dataType>> tcValues(rowWindowOffset.back(), std::vector<dataType>(ROW_WINDOW * COL_WINDOW, 0));

		for (vint i = 0; i < rowWindowOffset.size() - 1; ++i) {
			vint left_tc = rowWindowOffset[i];
			vint right_tc = rowWindowOffset[i + 1];

			for (vint tc_ind = left_tc; tc_ind < right_tc; ++tc_ind) {
				vint left_nnz = tcOffset[tc_ind];
				vint right_nnz = tcOffset[tc_ind + 1];
				for (vint nnz_id = left_nnz; nnz_id < right_nnz; ++nnz_id) {
					vint col_ind = sparseA2B[COL_WINDOW * tc_ind + tcLocalId[nnz_id] % COL_WINDOW];
					tcCols[tc_ind][tcLocalId[nnz_id]] = col_ind;
					tcValues[tc_ind][tcLocalId[nnz_id]] = data[nnz_id];
				}
			}
		}

		// 打印TC块
		if(printMsg){
			for (vint i = 0; i < rowWindowOffset.size() - 1; ++i) {
				vint left_tc = rowWindowOffset[i];
				vint right_tc = rowWindowOffset[i + 1];
				std::cout << "window = " << i << "\n";
				for (vint tc_ind = left_tc; tc_ind < right_tc; ++tc_ind) {
					std::cout << "TC Block = " << tc_ind << "\n";
					std::cout << "  vals         cols\n";
					const std::vector<vint>& tc_col_block = tcCols[tc_ind];
					const std::vector<dataType>& tc_val_block = tcValues[tc_ind];
					for (vint h = 0; h < ROW_WINDOW; ++h) {
						for (vint w = 0; w < COL_WINDOW; ++w) {
							std::cout << tc_val_block[h * COL_WINDOW + w] << " ";
						}
						std::cout << "    ";
						for (vint w = 0; w < COL_WINDOW; ++w) {
							std::cout << tc_col_block[h * COL_WINDOW + w] << " ";
						}
						std::cout << "\n";
					}
					std::cout << "\n";
				}
				std::cout << "\n";
			}
		}

	}

	void show(){

		std::cout << "\n========== ME-CTF =========:\n";
		std::cout << "rowWindowOffset:" << std::endl;
		for(auto iter = rowWindowOffset.begin(); iter != rowWindowOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;

		std::cout << "tcOffset:" << std::endl;
		for(auto iter = tcOffset.begin(); iter != tcOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;

		std::cout << "tcLocalId:" << std::endl;
		for(auto iter = tcLocalId.begin(); iter != tcLocalId.end(); iter++){
			std::cout << unsigned(*iter) << " ";
		}
		std::cout << std::endl;

		std::cout << "sparseA2B:" << std::endl;
		for(auto iter = sparseA2B.begin(); iter != sparseA2B.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;

		std::cout << "data:" << std::endl;
		for(auto iter = data.begin(); iter != data.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
	}

	void printSize(){

		long long s1 = rowWindowOffset.size() * sizeof(vint);
		long long s2 =  tcOffset.size() * sizeof(vint);
		long long s3 = tcLocalId.size() * sizeof(uint8_t);
		// long long s4 = sparseA2B.size()*sizeof(vint);
		// long long s5 = data.size()*sizeof(dataType);
		long long total = s1 + s2 + s3 ;

		std::cout << "\n========== size of ME-CTF =========\n";
		std::cout << "total size(window+tcOff+local):" << total << " bytes" << std::endl;
		std::cout << "rowWindowOffset: " << s1 << " bytes" << std::endl;
		// std::cout << "tcOffset: " << s2 << " bytes"  << std::endl;
		// std::cout << "tcLocalId: " << s3 << " bytes"  << std::endl;
		// std::cout << "sparseA2B: " << s4 << " bytes"  << std::endl;
		// std::cout << "data: " << s5 << " bytes"  << std::endl;

		std::cout << std::endl;
	}
};

template <class dataType>
struct METCFBit {
	std::vector<BIT_TYPE>       rowWindowOffsetBit;
	std::vector<TCLOCAL_TYPE>   tcLocalBit;
	std::vector<vint>           sparseA2B;
	std::vector<dataType>       data;

	void padListLength(std::vector<vint>& lst) {
		vint current_length = lst.size();
		vint remainder = current_length % COL_WINDOW;
		if (remainder != 0) {
			vint padding_size = COL_WINDOW - remainder;
			lst.insert(lst.end(), padding_size, UINT32_MAX);
		}
	}

	void METCF2METCFBit(const METCF<dataType>& metcf) {

		rowWindowOffsetBit.resize(metcf.rowWindowOffset.back());
		std::fill(rowWindowOffsetBit.begin(), rowWindowOffsetBit.end(), true);
		for (vint ind : metcf.rowWindowOffset) {
			if (ind != 0) {
				rowWindowOffsetBit[ind - 1] = false;
			}
		}

		vint tc_tmp_len = ROW_WINDOW * COL_WINDOW == 64 ? 1 : 2;
		// vint tc_tmp_len = 1;
		tcLocalBit.resize(metcf.rowWindowOffset.back() * tc_tmp_len, 0);
		for (vint i = 0; i < metcf.rowWindowOffset.back(); ++i) {
			for (vint ind = metcf.tcOffset[i]; ind < metcf.tcOffset[i + 1]; ++ind) {
				// tcLocalBit[i * ROW_WINDOW * COL_WINDOW + metcf.tcLocalId[ind]] = true;
				if (tc_tmp_len == 1){
					uint64_t local_idx_mask = 1ULL << metcf.tcLocalId[ind];
					tcLocalBit[i] |= local_idx_mask;
				} else if (tc_tmp_len == 2){
					uint8_t local_idx = metcf.tcLocalId[ind];
					if(local_idx < 64){
						uint64_t local_idx_mask = 1ULL << local_idx; // TC块内顺序从右向左看
						tcLocalBit[i * 2] |= local_idx_mask;
					}else{
						uint64_t local_idx_mask = 1ULL << (local_idx - 64);
						tcLocalBit[i * 2 + 1] |= local_idx_mask;
					}
				} else {
					std::cout << "ERROR: in file class.h-CSR2BME(): TC shape is not surported!!!" << std::endl;
				}
			}
		}

		sparseA2B = metcf.sparseA2B;
		data = metcf.data;
	}

	// // readMETCFBit(std::vector<std::vector<vint>>& tcCols, std::vector<std::vector<dataType>>& tcValues) {
	// void printTCblock(bool printMsg = false) {
	//     std::vector<std::vector<vint>> tcCols(rowWindowOffsetBit.size(), std::vector<vint>(ROW_WINDOW * COL_WINDOW, 0));
	//     std::vector<std::vector<dataType>> tcValues(rowWindowOffsetBit.size(), std::vector<dataType>(ROW_WINDOW * COL_WINDOW, 0));

	//     vint val_idx = 0;
	//     for (vint tc_ind = 0; tc_ind < rowWindowOffsetBit.size(); ++tc_ind) {
	//         vint lb = tc_ind * ROW_WINDOW * COL_WINDOW;
	//         vint hb = (tc_ind + 1) * ROW_WINDOW * COL_WINDOW;

	//         for (vint index = 0; index < ROW_WINDOW * COL_WINDOW; ++index) {
	//             if (tcLocalBit[lb + index]) {
	//                 vint col_ind = sparseA2B[COL_WINDOW * tc_ind + index % COL_WINDOW];
	//                 tcCols[tc_ind][index] = col_ind;
	//                 tcValues[tc_ind][index] = data[val_idx++];
	//             }
	//         }
	//     }

	//     // 打印TC块
	//     if(printMsg){
	//         vint window_ind = 0;
	//         for (vint tc_ind = 0; tc_ind < rowWindowOffsetBit.size(); ++tc_ind) {
	//             std::cout << "window = " << window_ind << std::endl;

	//             if(rowWindowOffsetBit[tc_ind] == 0){
	//                 window_ind++;
	//             }

	//             std::cout << "TC Block = " << tc_ind << "\n";
	//             std::cout << "  vals         cols\n";
	//             const std::vector<dataType>& tc_val_block = tcValues[tc_ind];
	//             const std::vector<vint>& tc_col_block = tcCols[tc_ind];

	//             for (vint h = 0; h < ROW_WINDOW; ++h) {
	//                 for (vint w = 0; w < COL_WINDOW; ++w) {
	//                     std::cout << tc_val_block[h * COL_WINDOW + w] << " ";
	//                 }
	//                 std::cout << "    ";
	//                 for (vint w = 0; w < COL_WINDOW; ++w) {
	//                     std::cout << tc_col_block[h * COL_WINDOW + w] << " ";
	//                 }
	//                 std::cout << "\n";
	//             }
	//             std::cout << std::endl;
	//         }
	//     }

	// }

	void show() {

		// std::cout << "\n========== ME-CTF-Bit =========:\n";
		// std::cout << "rowWindowOffsetBit:" << std::endl;
		// for(auto iter = rowWindowOffsetBit.begin(); iter != rowWindowOffsetBit.end();iter++){
		//     std::cout << *iter << " ";
		// }
		// std::cout << std::endl;

		// // // bool时的show
		// // std::cout << "tcLocalBit:" << std::endl;
		// // vint cnt = 0; vint cnt2 = 0;
		// // vint num_tc_block = 0;
		// // for(auto iter = tcLocalBit.begin(); iter != tcLocalBit.end(); iter++){
		// //     ++cnt;
		// //     std::cout << *iter << " ";
		// //     if((cnt % COL_WINDOW == 0) && cnt != 0) {std::cout << std::endl;}
		// //     if((cnt % (COL_WINDOW * ROW_WINDOW) == 0) && cnt != 0) {
		// //         std::cout << " == end_tc_block" << std::endl; 
		// //         if(rowWindowOffsetBit[num_tc_block++] == false) {
		// //             std::cout << " == end_whole_thread_block" << std::endl;
		// //         }
		// //         cnt = 0;
		// //     }
		// // }


		// std::cout << "tcLocalBit:" << std::endl;
		// vint cnt = 0;
		// std::cout << "tcLocalBit.size() = " << tcLocalBit.size() <<std::endl;
		for(auto iter = tcLocalBit.begin(); iter != tcLocalBit.end();iter++){
			std::cout << std::bitset<64>(*iter) << std::endl;
			std::cout << (*iter) << std::endl;
		}
		
		// std::cout << std::endl;
		// std::cout << std::endl;
		// std::cout << std::endl;
		// std::cout << std::endl;

		std::cout << "sparseA2B:" << std::endl;
		for(auto iter = sparseA2B.begin(); iter != sparseA2B.end();iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;

		// std::cout << "data:" << std::endl;
		// cnt = 0;
		for(auto iter = data.begin(); iter != data.end();iter++){
			// std::cout << *iter << " ";
			std::cout << *iter << " ";
			// ++cnt;
			// if(cnt == COL_WINDOW) {std::cout << std::endl; cnt = 0;}
		}
		std::cout << std::endl;
	}

	long long printSize(){

		// std::vector<bool> rowWindowOffsetBit;
		// std::vector<uint64_t> tcLocalBit;
		// std::vector<vint> sparseA2B;
		// std::vector<dataType> data;

		long long s1 = (rowWindowOffsetBit.size() + 7) / 8;
		long long s2 = tcLocalBit.size() * sizeof(uint64_t);
		long long s3 = sparseA2B.size() * sizeof(int);
		// long long s4 = data.size() * sizeof(dataType);
		long long total = s1 + s2 + s3;

		std::cout << "\n========== size of ME-CTF-bit =========\n";
		std::cout << "total size(without data): " << total << " bytes" << std::endl;
		std::cout << "rowWindowOffsetBit: " << s1 << " bytes" << std::endl;
		std::cout << "tcLocalBit: " << s2 << " bytes"  << std::endl;
		std::cout << "sparseA2B: " << s3 << " bytes"  << std::endl;
		// std::cout << "data: " << s4 << " bytes"  << std::endl;

		std::cout << std::endl;
		return total;
	}
};

// BME: Balanced and Memory-Efficient format
template <class dataType>
struct BME {
	std::vector<vint> groupOffset; // 每个group首个TC块的偏移量（是全局第几个TC块）
	std::vector<vint> tcOffset; // 每个TC块首个非零元的偏移量
	std::vector<vint> rowIndices; // 每个group的首行id
	std::vector<TCLOCAL_TYPE> tcLocalBit; // TC块大小位8*8时：长度等于TC块个数，每个数字代表一个TC块的bitmap；16*8时，每两个数字代表一个TC块
	std::vector<vint> sparseA2B;
	std::vector<dataType> data;

	void padListLength(std::vector<vint>& lst) {
		vint current_length = lst.size();
		vint remainder = current_length % COL_WINDOW;
		if (remainder != 0) {
			vint padding_size = COL_WINDOW - remainder;
			lst.insert(lst.end(), padding_size, UINT32_MAX);
		}
	}

	void CSR2BME(const CSR<dataType>& csr) {
		vint num_nodes = csr.rows;
		vint total_tc_num = 0;
		groupOffset.push_back(0);
		for (vint iter = 0; iter < num_nodes; iter += ROW_WINDOW) {
			// vint windowId = iter / ROW_WINDOW;
			vint block_start = csr.row_ptr[iter];
			vint block_end = csr.row_ptr[std::min(iter + ROW_WINDOW, num_nodes)];

			std::vector<vint> neighbor_window(csr.col_idx.begin() + block_start, csr.col_idx.begin() + block_end);
			std::sort(neighbor_window.begin(), neighbor_window.end());
			std::vector<vint> unique_edges;
			std::unique_copy(neighbor_window.begin(), neighbor_window.end(), std::back_inserter(unique_edges));

			std::unordered_map<vint, vint> clean_edges2col;
			for (vint i = 0; i < unique_edges.size(); ++i) {
				clean_edges2col[unique_edges[i]] = i;
			}
			
			padListLength(unique_edges);
			sparseA2B.insert(sparseA2B.end(), unique_edges.begin(), unique_edges.end());

			vint window_tc_num = (unique_edges.size() + COL_WINDOW - 1) / COL_WINDOW;
			vint window_group_num = (window_tc_num + GROUP_LEN - 1) / GROUP_LEN;
			tcOffset.resize(tcOffset.size() + window_tc_num, 0);
			rowIndices.resize(rowIndices.size() + window_group_num, iter);

			std::vector<std::vector<dataType>> datatmp(window_tc_num);
			vint tc_tmp_len = ROW_WINDOW * COL_WINDOW == 64 ? 1 : 2; // ROW_WINDOW * COL_WINDOW = 64 | 128
			// vint tc_tmp_len = 1;
			std::vector<uint64_t> tcLocalBittmp(window_tc_num * tc_tmp_len, 0);
			for (vint r = iter; r < std::min(iter + ROW_WINDOW, num_nodes); ++r) {
				for (vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id) {
					vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
					vint tc_idx = total_tc_num + c_idx / COL_WINDOW;
					tcOffset[tc_idx]++;
					if (tc_tmp_len == 1){
						uint64_t local_idx_mask = 1ULL << ((r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW);
						tcLocalBittmp[c_idx / COL_WINDOW] |= local_idx_mask;
					} else if(tc_tmp_len == 2){
						uint64_t local_idx = (r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW;
						if (local_idx < 64){
							uint64_t local_idx_mask = 1ULL << local_idx; // TC块内顺序从右向左看
							tcLocalBittmp[c_idx / COL_WINDOW * 2] |= local_idx_mask;
						} else{
							uint64_t local_idx_mask = 1ULL << (local_idx-64);
							tcLocalBittmp[c_idx / COL_WINDOW * 2 + 1] |= local_idx_mask;
						}
					} else {
						std::cout << "ERROE: in file class.h-CSR2BME(): TC shape is not surportable!!" << std::endl;
					}
					datatmp[c_idx / COL_WINDOW].push_back(csr.data[nnz_id]);
				}
			}
			tcLocalBit.insert(tcLocalBit.end(), tcLocalBittmp.begin(), tcLocalBittmp.end());
			groupOffset.insert(groupOffset.end(), window_tc_num/GROUP_LEN, GROUP_LEN);
			if(window_tc_num%GROUP_LEN!=0) groupOffset.push_back(window_tc_num%GROUP_LEN);

			for (vint i = 0; i < datatmp.size(); ++i) {
				data.insert(data.end(), datatmp[i].begin(), datatmp[i].end());
			}
			total_tc_num += window_tc_num;
		}
	   
		tcOffset.insert(tcOffset.begin(), 0);
		for(vint i = 0; i < tcOffset.size()-1; i++){
			tcOffset[i+1] += tcOffset[i];
		}

		for(vint i = 0; i < groupOffset.size()-1; i++){
			groupOffset[i+1] += groupOffset[i];
		}
	}

	void show(){
		/*
			std::vector<vint> groupOffset; // 每个group首个TC块的偏移量
			std::vector<vint> tcOffset; // 每个TC块首个非零元的偏移量
			std::vector<vint> rowIndices; // 每个gorup的首行id
			std::vector<uint64_t> tcLocalBit; // TC块大小位8*8时：长度等于TC块个数，每个数字代表一个TC块的bitmap；16*8时，每两个数字代表一个TC块
			std::vector<vint> sparseA2B;
			std::vector<dataType> data;
		*/

		std::cout << "\n========== balance-Bit =========:\n";

		std::cout << "groupOffset:" << groupOffset.size() << std::endl;
		for(auto iter = groupOffset.begin(); iter != groupOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "tcOffset:" << tcOffset.size() << std::endl;
		for(auto iter = tcOffset.begin(); iter != tcOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "rowIndices:" << rowIndices.size() << std::endl;
		for(auto iter = rowIndices.begin(); iter != rowIndices.end();iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "tcLocalBit:" << std::endl;
		// vint cnt = 0;
		std::cout << "tcLocalBit.size() = " << tcLocalBit.size() <<std::endl;
		for(auto iter = tcLocalBit.begin(); iter != tcLocalBit.end(); iter++){
			std::cout << std::bitset<64>(*iter) << std::endl;
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "sparseA2B:" << std::endl;
		vint cnt = 0;
		for(auto iter = sparseA2B.begin(); iter != sparseA2B.end(); iter++){
			std::cout << *iter << " ";
			++cnt;
			if(cnt == COL_WINDOW) {std::cout << std::endl; cnt = 0;}
		}

		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "data:" << std::endl;

		for(vint i = 0; i < groupOffset.size()-1; i++){
			for(vint nnz_id = groupOffset[i]; nnz_id < groupOffset[i+1]; nnz_id++){
				std::cout << data[nnz_id] << " ";
			}
			std::cout << std::endl;
		}
	}

	long long printSize(){
		long long s1 = groupOffset.size() * sizeof(int); // 应该只能计算用int存储的空间
		long long s2 = rowIndices.size() * sizeof(int); 
		long long s3 = tcLocalBit.size() * sizeof(uint64_t);
		long long s4 = sparseA2B.size() * sizeof(int);
		// long long s5 = data.size()*sizeof(dataType);
		long long total = s1 + s2 + s3 + s4;

		std::cout << "\n========== size of balance-bit(ours) =========\n";
		std::cout << "total size(without data array): " << total << " bytes" << std::endl;
		std::cout << "groupOffset: " << s1 << " bytes" << std::endl;
		std::cout << "rowIndices: " << s2 << " bytes" << std::endl;
		std::cout << "tcLocalBit: " << s3 << " bytes"  << std::endl;
		std::cout << "sparseA2B: " << s4 << " bytes"  << std::endl;
		// std::cout << "data: " << s5 << " bytes"  << std::endl;

		std::cout << std::endl;
		return total;
	}
};

// Adaptive BME: Adaptive Balanced and Memory-Efficient format
template <class dataType>
struct AdpBME {
	std::vector<vint> groupOffset; // 每个group首个TC块的偏移量（是全局第几个TC块）len = group num
	std::vector<vint> tcOffset;    // 每个TC块首个非零元的偏移量 len = tc num
	std::vector<vint> rowIndices;  // 每个TC块的首行id len = tc num
	std::vector<TCLOCAL_TYPE> tcLocalBit; // TC块大小位8*8时：长度等于TC块个数，每个数字代表一个TC块的bitmap；16*8时，每两个数字代表一个TC块
	std::vector<vint> sparseA2B;   // len = nnz num
	std::vector<dataType> data;    // len = nnz num

	void padListLength(std::vector<vint>& lst) {
		vint current_length = lst.size();
		vint remainder = current_length % COL_WINDOW;
		if (remainder != 0) {
			vint padding_size = COL_WINDOW - remainder;
			lst.insert(lst.end(), padding_size, UINT32_MAX);
		}
	}

	void CSR2AdpBME(const CSR<dataType>& csr, vint t_load, vint t_comp, vint t_write, vint target_num = TARGET_NUM) {

		vint num_nodes = csr.rows;
		vint total_tc_num = 0;
		
		// vint t_tc = t_load + t_comp;
		vint target = target_num * (t_load + t_comp) + t_write;
		vint current_group_time = 0;
		vint current_group_size = 0;
		groupOffset.push_back(0);

		for (vint iter = 0; iter < num_nodes; iter += ROW_WINDOW) {
			// vint windowId = iter / ROW_WINDOW;
			vint block_start = csr.row_ptr[iter];
			vint block_end = csr.row_ptr[std::min(iter + ROW_WINDOW, num_nodes)];

			std::vector<vint> neighbor_window(csr.col_idx.begin() + block_start, csr.col_idx.begin() + block_end);
			std::sort(neighbor_window.begin(), neighbor_window.end());
			std::vector<vint> unique_edges;
			std::unique_copy(neighbor_window.begin(), neighbor_window.end(), std::back_inserter(unique_edges));

			std::unordered_map<vint, vint> clean_edges2col;
			for (vint i = 0; i < unique_edges.size(); ++i) {
				clean_edges2col[unique_edges[i]] = i;
			}
			
			padListLength(unique_edges);
			sparseA2B.insert(sparseA2B.end(), unique_edges.begin(), unique_edges.end());

			vint window_tc_num = (unique_edges.size() + COL_WINDOW - 1) / COL_WINDOW;
			// vint window_group_num = (window_tc_num + GROUP_LEN - 1) / GROUP_LEN;
			tcOffset.resize(tcOffset.size() + window_tc_num, 0);
			rowIndices.resize(rowIndices.size() + window_tc_num, iter);

			vint additional_time = 0;
			for(vint i = 0; i < window_tc_num; i++){
				if(i == 0){
					additional_time = t_load + t_comp + t_write;
				}else{
					additional_time = t_load + t_comp ;
				}

				if (current_group_size > 0 && current_group_time + additional_time > target) {
					groupOffset.push_back(current_group_size);  // 如果加上当前tc的需要时间，会超过 target，所以把当前tc划分到下一组的开始
					current_group_time = 0;
					current_group_size = 0;
				}

				current_group_time += additional_time;
				current_group_size++;
			}

			std::vector<std::vector<dataType>> datatmp(window_tc_num);
			vint tc_tmp_len = ROW_WINDOW * COL_WINDOW == 64 ? 1 : 2; // ROW_WINDOW * COL_WINDOW = 64 | 128
			std::vector<uint64_t> tcLocalBittmp(window_tc_num * tc_tmp_len, 0);
			for (vint r = iter; r < std::min(iter + ROW_WINDOW, num_nodes); ++r) {
				for (vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id) {
					vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
					vint tc_idx = total_tc_num + c_idx / COL_WINDOW;
					tcOffset[tc_idx]++;
					if (tc_tmp_len == 1){
						uint64_t local_idx_mask = 1ULL << ((r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW);
						tcLocalBittmp[c_idx / COL_WINDOW] |= local_idx_mask;
					} else if(tc_tmp_len == 2){
						uint64_t local_idx = (r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW;
						if (local_idx < 64){
							uint64_t local_idx_mask = 1ULL << local_idx; // TC块内顺序从右向左看
							tcLocalBittmp[c_idx / COL_WINDOW * 2] |= local_idx_mask;
						} else{
							uint64_t local_idx_mask = 1ULL << (local_idx-64);
							tcLocalBittmp[c_idx / COL_WINDOW * 2 + 1] |= local_idx_mask;
						}
					} else {
						std::cout << "ERROE: in file class.h-CSR2BME(): TC shape is not surportable!!" << std::endl;
					}
					datatmp[c_idx / COL_WINDOW].push_back(csr.data[nnz_id]);
				}
			}
			tcLocalBit.insert(tcLocalBit.end(), tcLocalBittmp.begin(), tcLocalBittmp.end());
			// groupOffset.insert(groupOffset.end(), window_tc_num/GROUP_LEN, GROUP_LEN);
			// if(window_tc_num%GROUP_LEN!=0) groupOffset.push_back(window_tc_num%GROUP_LEN);

			for (vint i = 0; i < datatmp.size(); ++i) {
				data.insert(data.end(), datatmp[i].begin(), datatmp[i].end());
			}
			total_tc_num += window_tc_num;
		}

		if(current_group_size != 0 ){
			groupOffset.push_back(current_group_size);
		}
		for(vint i = 0; i < groupOffset.size()-1; i++){
			groupOffset[i+1] += groupOffset[i];
		}    


		tcOffset.insert(tcOffset.begin(), 0);
		for(vint i = 0; i < tcOffset.size()-1; i++){
			tcOffset[i+1] += tcOffset[i];
		}
	}

	void show(){
		std::cout << "\n========== balance-Bit =========:\n";

		std::cout << "groupOffset:" << std::endl;
		for(auto iter = groupOffset.begin(); iter != groupOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "tcOffset:" << std::endl;
		for(auto iter = tcOffset.begin(); iter != tcOffset.end(); iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		std::cout << "rowIndices:" << std::endl;
		for(auto iter = rowIndices.begin(); iter != rowIndices.end();iter++){
			std::cout << *iter << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		// std::cout << "tcLocalBit:" << std::endl;
		// // vint cnt = 0;
		// std::cout << "tcLocalBit.size() = " << tcLocalBit.size() <<std::endl;
		// for(auto iter = tcLocalBit.begin(); iter != tcLocalBit.end(); iter++){
		//     std::cout << std::bitset<64>(*iter) << std::endl;
		// }
		// std::cout << std::endl;
		// std::cout << std::endl;

		// std::cout << "sparseA2B:" << std::endl;
		// vint cnt = 0;
		// for(auto iter = sparseA2B.begin(); iter != sparseA2B.end(); iter++){
		//     std::cout << *iter << " ";
		//     ++cnt;
		//     if(cnt == COL_WINDOW) {std::cout << std::endl; cnt = 0;}
		// }

		// std::cout << std::endl;
		// std::cout << std::endl;
		// std::cout << "data:" << std::endl;

		// for(vint i = 0; i < groupOffset.size()-1; i++){
		//     for(vint nnz_id = groupOffset[i]; nnz_id < groupOffset[i+1]; nnz_id++){
		//         std::cout << data[nnz_id] << " ";
		//     }
		//     std::cout << std::endl;
		// }
	}

	long long printSize(bool print = true){
		long long s1 = groupOffset.size() * sizeof(int);
		long long s2 = tcOffset.size() * sizeof(int);
		long long s3 = rowIndices.size() * sizeof(int); 
		long long s4 = tcLocalBit.size() * sizeof(TCLOCAL_TYPE);
		long long s5 = sparseA2B.size() * sizeof(int);
		long long total = s1 + s2 + s3 + s4 + s5;

		if(print){
			std::cout << "\n========== size of BME (ours) =========\n";
			std::cout << "total size(without data array): " << total << " bytes" << std::endl;
			std::cout << "groupOffset: " << s1 << " bytes" << std::endl;
			std::cout << "tcOffset: " << s2 << " bytes" << std::endl;
			std::cout << "rowIndices: " << s3 << " bytes"  << std::endl;
			std::cout << "tcLocalBit: " << s4 << " bytes"  << std::endl;
			std::cout << "sparseA2B: " << s5 << " bytes"  << std::endl;            
			std::cout << std::endl;
		}

		return total;
	}
};

#endif // __CLASS_H__
