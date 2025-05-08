#ifndef ASPT_SDDMM_H
#define ASPT_SDDMM_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <xmmintrin.h>
// #include "mkl.h"
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include<math.h>
#include<iostream>
// using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))

#if DOUBLE == 1
    #define FTYPE double
#else
	#define FTYPE float
#endif

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)
#define THRESHOLD (16*1)
#define BH (128*1)
#define LOG_BH (7)
#define BW (128*1)
#define MIN_OCC (BW*3/4)
//#define MIN_OCC (-1)
#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
#define NTHREAD (128)
#define SC_SIZE (2048)

// #define ITER (128)

void aspt_preprocess_cpu(
	int * row_ptr, int * col_idx0, FTYPE * val0, 
	int * col_idx, FTYPE * val, 
	int n, int nnz, 
	int nr, int npanel, 
	double *avg, double *vari,
	int *special, int *special2, int *special_p,
	int *mcsr_e, int *mcsr_cnt, int *mcsr_chk);

void aspt_sddmm_cpu(
	int * col_idx, FTYPE * val, 
	FTYPE * A, FTYPE * B, 
	FTYPE * val_out, 
	int middle, 
	int nr, double vari, 
	int *special, int *special2, int special_p, 
	int *mcsr_e, int *mcsr_cnt);

#endif // ASPT_SDDMM_H