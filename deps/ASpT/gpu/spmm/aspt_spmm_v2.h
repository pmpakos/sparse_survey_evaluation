#ifndef ASPT_SPMM_V2_H
#define ASPT_SPMM_V2_H

#include <iostream>
#include <vector>
#include <algorithm>

#include "shfl_fix.h"

#define ERR_INFO(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }

#define CMP_SWP(t1,_a,_b,t2,_c,_d) if(_a>_b)  {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}
#define EQL_SWP(t1,_a,_b,t2,_c,_d) if(_a!=_b) {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}
#define     SWP(t1,_a,_b,t2,_c,_d)            {t1 _t=_a;_a=_b;_b=_t; t2 _s=_c;_c=_d;_d=_s;}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))

#define SEGBIN_NUM 13

#define FTYPE float
#define STYPE int

#define MFACTOR (32)
#define LOG_MFACTOR (5)

#define BSIZE (1024/1)
#define BF (BSIZE/32)


#define INIT_GRP (10000000)
#define INIT_LIST (-2147483648)
#define THRESHOLD (8*2)
#define BH (128/1)
#define BW (128/1)
#define MIN_OCC (BW*3/4)

#define SBSIZE (1024/8)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)

#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define SC_SIZE (2048)

#define MCSR_CNT_SIZE (1024)
#define LIST_CANDI (1024*4)

#define MCSR_CNT_TBSIZE (1024)

/************************************************************************/
/* FUNCTION DECLARATIONS */

/************************** BB_EXCH.H (start)  **************************/
// Exchange intersection for 1 keys.
__device__ inline void exch_intxn(int &k0, int &v0, int mask, const int bit);
// Exchange intersection for 2 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &v0, int &v1, int mask, const int bit);
// Exchange intersection for 4 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit);
// Exchange intersection for 8 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit);
// Exchange intersection for 16 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit);
// Exchange intersection for 32 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &k16, int &k17, int &k18, int &k19, int &k20, int &k21, int &k22, int &k23, int &k24, int &k25, int &k26, int &k27, int &k28, int &k29, int &k30, int &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit);
// Exchange parallel for 1 keys.
__device__ inline void exch_paral(int &k0, int &v0, int mask, const int bit);
// Exchange parallel for 2 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &v0, int &v1, int mask, const int bit);
// Exchange parallel for 4 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit);
// Exchange parallel for 8 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit);
// Exchange parallel for 16 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit);
// Exchange parallel for 32 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &k16, int &k17, int &k18, int &k19, int &k20, int &k21, int &k22, int &k23, int &k24, int &k25, int &k26, int &k27, int &k28, int &k29, int &k30, int &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit);
/*************************** BB_EXCH.H (end)  ***************************/

/************************ BB_COMPUT_S.H (start)  ************************/
__device__  int find_kth3(int* a, int aCount, int* b, int bCount, int diag);

__global__ void gen_copy(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   256   1       2     false  true      true */
__global__ void gen_bk256_wp2_tc1_r2_r2_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2       2     false  true      true */
__global__ void gen_bk128_wp2_tc2_r3_r4_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       2     false  true      true */
__global__ void gen_bk128_wp2_tc4_r5_r8_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       4      true  true      true */
__global__ void gen_bk128_wp4_tc4_r9_r16_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       8      true  true      true */
__global__ void gen_bk128_wp8_tc4_r17_r32_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4      16      true  true      true */
__global__ void gen_bk128_wp16_tc4_r33_r64_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   256  16       8      true  true      true */
__global__ void gen_bk256_wp8_tc16_r65_r128_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf subwarp coalesced quiet real_kern */
/*   256   8      32      true  true      true */
__global__ void gen_bk256_wp32_tc8_r129_r256_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf1 tcf2 quiet real_kern */
/*   128    2    4  true      true */
__global__ void gen_bk128_tc4_r257_r512_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf1 tcf2 quiet real_kern */
/*   256    2    4  true      true */
__global__ void gen_bk256_tc4_r513_r1024_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);
/* block tcf1 tcf2 quiet real_kern */
/*   512    2    4  true      true */
__global__ void gen_bk512_tc4_r1025_r2048_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length);

/************************* BB_COMPUT_S.H (end)  *************************/

/************************ BB_COMPUT_L.H (start)  ************************/
__device__ int binary_search(int *blk_stat, int bin_size, int gid, int blk_num);
__device__ inline int upper_power_of_two(int v);
__device__ inline int log2(int u);
__global__ void kern_get_num_blk_init(int *max_segsize, int *segs, int *bin, int *blk_stat, int n, int bin_size, int length, int workloads_per_block);
__global__ void kern_get_init_pos(int *blk_stat, int *blk_innerid, int *blk_seg_start, int blk_num, int bin_size);

__global__ void kern_block_sort(int *key, int *val, int *keyB, int *valB, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n);

__global__ void kern_get_num_blk(int *segs, int *bin, int *blk_stat, int n, int bin_size, int length, int workloads_per_block);

__global__ void kern_block_merge(int *keys, int *vals, int *keysB, int *valsB, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n, int stride);

__global__ void kern_copy(int *srck, int *srcv, int *dstk, int *dstv, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n, int res);

int gen_grid_kern_r2049(int *keys_d, int *vals_d, int *keysB_d, int *valsB_d, int n, int *segs_d, int *bin_d, int bin_size, int length);
/************************* BB_COMPUT_L.H (end)  *************************/

/************************** BB_BIN.H (start)   **************************/
void bb_bin(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, const int length, const int n, int *h_bin_counter);
__global__ void bb_bin_histo(int *d_bin_counter, const int *d_segs, int length, int n);
__global__ void bb_bin_group(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, int length, int n);
/*************************** BB_BIN.H (end)  ***************************/

/************************* BB_SEGSORT.H (start)  ************************/
int bb_segsort(int *keys_d, int *vals_d, int n,  int *d_segs, int length);

void show_d(int *arr_d, int n, std::string prompt);
/************************  BB_SEGSORT.H (end)  ************************/

/*************************** MAIN.H (start)  *************************/
__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_sparse_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout);
__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_sparse_v2l(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout);

__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_sparse_v2h(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, int *special, int *special2);

__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_ssparse(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout);

__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_dense_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, int *baddr, int *saddr);

__global__ void dense_block_detect(int *csr_v, int *mcsr_chk, int *csr_e0, int *flag);
__global__ void simple_mcsr_cnt(int npanel, int *mcsr_cnt);
__global__ void csr_pivot_gen(int npanel, int *csr_v, int *csr_pivot);
__global__ void csr_pnt_gen(int ne, int *csr_e0, int *key, STYPE *key2, int *val);
__global__ void mcsr_cnt_calc(int *csr_pivot, int *key, int *mcsr_cnt, int *mcsr_chk);
__global__ void key2_marking(int *csr_pivot, int *key, STYPE *key2, int *val, int *mcsr_cnt, int *mcsr_list, int *baddr, int *saddr, int *mcsr_chk);
__global__ void fill_val(int ne, int *val);
__global__ void fill_mcsre(int *csr_v, int *mcsr_cnt, STYPE *key2, int *mcsr_e, int *rmv);
__global__ void porting(int ne, int *val, int *csr_e0, FTYPE *csr_ev0, int *csr_e, FTYPE *csr_ev);
__global__ void cal_vari(int nr, double avg, int *mcsr_cnt, int *mcsr_e, double *vari, int *special_bb);
__global__ void make_special(int *mcsr_cnt, int *mcsr_e, int *special, int *special2, int *scnt);

/**************************  MAIN.H (end)  **************************/
/************************************************************************/

#endif // ASPT_SPMM_V2_H
