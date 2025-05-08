#include "aspt_spmm.h"

void aspt_preprocess_cpu(
	int * row_ptr, int * col_idx0, FTYPE * val0, 
	int * col_idx, FTYPE * val, 
	int n, int nnz, 
	int nr, int npanel, 
	double *avg, double *vari,
	int *special, int *special2, int *special_p,
	int *mcsr_e, int *mcsr_cnt, int *mcsr_chk)
{
	int nthreads = omp_get_max_threads();

	// special = (int *)malloc(sizeof(int)*nnz);
	// special2 = (int *)malloc(sizeof(int)*nnz);
	memset(special, 0, sizeof(int)*nnz);
	memset(special2, 0, sizeof(int)*nnz);

	// mcsr_cnt = (int *)malloc(sizeof(int)*(npanel+1));
	// mcsr_chk = (int *)malloc(sizeof(int)*(npanel+1));
	// mcsr_e = (int *)malloc(sizeof(int)*nnz); // reduced later
	memset(mcsr_cnt, 0, sizeof(int)*(npanel+1));
	memset(mcsr_chk, 0, sizeof(int)*(npanel+1));
	memset(mcsr_e, 0, sizeof(int)*nnz);	

	int bv_size = CEIL(n, 32);
	unsigned int **bv = (unsigned int **)malloc(sizeof(unsigned int *)*nthreads);
	for(int i=0;i<nthreads;i++) 
		bv[i] = (unsigned int *)malloc(sizeof(unsigned int)*bv_size);
	int **col_idx1 = (int **)malloc(sizeof(int *)*2);
	short **coo = (short **)malloc(sizeof(short *)*2);
	for(int i=0;i<2;i++) {
		col_idx1[i] = (int *)malloc(sizeof(int)*nnz);
		coo[i] = (short *)malloc(sizeof(short)*nnz);
	}

	struct timeval starttime0, starttime, endtime;
	gettimeofday(&starttime0, NULL);

	// filtering(WILL)
	//memcpy(col_idx1[0], col_idx0, sizeof(int)*nnz);
	#pragma omp parallel for schedule(dynamic, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
			for(int j=row_ptr[i]; j<row_ptr[i+1]; j++) {
				col_idx1[0][j] = col_idx0[j];
			}
		}
	}

	char scr_pad[NTHREAD][SC_SIZE];

	gettimeofday(&starttime, NULL);

	#pragma omp parallel for schedule(dynamic, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		int tid = omp_get_thread_num();
		int i, j, t_sum=0;

		// coo generate and mcsr_chk
		memset(scr_pad[tid], 0, sizeof(char)*SC_SIZE);
		for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
			for(j=row_ptr[i]; j<row_ptr[i+1]; j++) {
				coo[0][j] = (i&(BH-1));
				int kk = (col_idx0[j]&(SC_SIZE-1));
				if(scr_pad[tid][kk] < THRESHOLD) {
					if(scr_pad[tid][kk] == THRESHOLD - 1) t_sum++;
					scr_pad[tid][kk]++;
				}
			}
		}

		if (t_sum < MIN_OCC) {
			mcsr_chk[row_panel] = 1;
			mcsr_cnt[row_panel+1] = 1;
			continue;
		} 

		// sorting(merge sort)
		int flag = 0;
		for(int stride = 1; stride <= BH/2; stride *= 2, flag=1-flag) {
			for(int pivot = row_panel*BH; pivot < (row_panel+1)*BH; pivot += stride*2) {
				int l1, l2;
				for(i = l1 = row_ptr[pivot], l2 = row_ptr[pivot+stride]; l1 < row_ptr[pivot+stride] && l2 < row_ptr[pivot+stride*2]; i++) {
					if(col_idx1[flag][l1] <= col_idx1[flag][l2]) {
						coo[1-flag][i] = coo[flag][l1];
						col_idx1[1-flag][i] = col_idx1[flag][l1++];
					}
					else {
						coo[1-flag][i] = coo[flag][l2];
						col_idx1[1-flag][i] = col_idx1[flag][l2++];	
					}
				}
				while(l1 < row_ptr[pivot+stride]) {
					coo[1-flag][i] = coo[flag][l1];
					col_idx1[1-flag][i++] = col_idx1[flag][l1++];
				}
				while(l2 < row_ptr[pivot+stride*2]) {
					coo[1-flag][i] = coo[flag][l2];
					col_idx1[1-flag][i++] = col_idx1[flag][l2++];
				}
			}
		}				

		int weight=1;

		// int cq=0;
		int cr=0;

		// dense bit extract (and mcsr_e making)
		for(i=row_ptr[row_panel*BH]+1; i<row_ptr[(row_panel+1)*BH]; i++) {
			if(col_idx1[flag][i-1] == col_idx1[flag][i]) weight++;
			else {
				if(weight >= THRESHOLD) {
					cr++;
				} //if(cr == BW) { cq++; cr=0;}
				weight = 1;
			}
		}
		//int reminder = (col_idx1[flag][i-1]&31);
		if(weight >= THRESHOLD) {
			cr++;
		} //if(cr == BW) { cq++; cr=0; }

		// TODO = occ control
		mcsr_cnt[row_panel+1] = CEIL(cr,BW)+1;
	}

	// prefix-sum
	for(int i=1; i<=npanel;i++) 
		mcsr_cnt[i] += mcsr_cnt[i-1];
	//mcsr_e[0] = 0;
	mcsr_e[BH * mcsr_cnt[npanel]] = nnz;

	double *avg0;
	avg0 = (double *)malloc(sizeof(double)*nthreads);
	memset(avg0, 0, sizeof(double)*nthreads);

	#pragma omp parallel for //schedule(dynamic, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		int tid = omp_get_thread_num();
		// printf("tid = %d, row_panel = %d, mcsr_chk[row_panel] = %d\n", tid, row_panel, mcsr_chk[row_panel]);
		if(mcsr_chk[row_panel] == 0) {
			int i, j;
			int flag = 0;
			int cq=0, cr=0;
			for(int stride = 1; stride <= BH/2; stride*=2, flag=1-flag);
			int base = (mcsr_cnt[row_panel]*BH);
			int mfactor = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
			int weight=1;

			// mcsr_e making
			for(i=row_ptr[row_panel*BH]+1; i<row_ptr[(row_panel+1)*BH]; i++) {
				if(col_idx1[flag][i-1] == col_idx1[flag][i]) weight++;
				else {
					int reminder = (col_idx1[flag][i-1]&31);
					if(weight >= THRESHOLD) {
						cr++;
						bv[tid][col_idx1[flag][i-1]>>5] |= (1<<reminder); 
						for(j=i-weight; j<=i-1; j++) {
							mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
						}
					} else {
						//bv[tid][col_idx1[flag][i-1]>>5] &= (~0 - (1<<reminder)); 
						bv[tid][col_idx1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder)); 
					} 
					if(cr == BW) { cq++; cr=0;}
					weight = 1;
				}
			}

			//fprintf(stderr, "inter : %d\n", i);

			int reminder = (col_idx1[flag][i-1]&31);
			if(weight >= THRESHOLD) {
				cr++;
				bv[tid][col_idx1[flag][i-1]>>5] |= (1<<reminder); 
				for(j=i-weight; j<=i-1; j++) {
					mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
				}
			} else {
				bv[tid][col_idx1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder)); 
			} 
			// reordering
			int delta = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
			int base0 = mcsr_cnt[row_panel]*BH;
			for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
				int base = base0+(i-row_panel*BH)*delta;
				int dpnt = mcsr_e[base] = row_ptr[i];
				for(int j=1;j<delta;j++) {
					mcsr_e[base+j] += mcsr_e[base+j-1];
				}
				int spnt=mcsr_e[mcsr_cnt[row_panel]*BH + (mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel])*(i - row_panel*BH + 1) - 1];

				avg0[tid] += row_ptr[i+1] - spnt; 
				for(j=row_ptr[i]; j<row_ptr[i+1]; j++) {
					int kk = col_idx0[j];
					if((bv[tid][kk>>5]&(1<<(kk&31)))) {
						col_idx[dpnt] = col_idx0[j];
						val[dpnt++] = val0[j];
					} else {
						col_idx[spnt] = col_idx0[j];
						val[spnt++] = val0[j];
					}
				}
			}
		} 
		else {
			int base0 = mcsr_cnt[row_panel]*BH;
			// printf("tid = %d, memcpy(&mcsr_e[base0], &row_ptr[row_panel*BH], sizeof(int)*BH); BEGIN\n", tid, row_panel, base0);
			memcpy(&mcsr_e[base0], &row_ptr[row_panel*BH], sizeof(int)*BH);
			// printf("tid = %d, memcpy(&mcsr_e[base0], &row_ptr[row_panel*BH], sizeof(int)*BH); END\n", tid, row_panel, base0);
			avg0[tid] += row_ptr[(row_panel+1)*BH] - row_ptr[row_panel*BH];
			int bidx = row_ptr[row_panel*BH];
			int bseg = row_ptr[(row_panel+1)*BH] - bidx;
			memcpy(&col_idx[bidx], &col_idx0[bidx], sizeof(int)*bseg);
			memcpy(&val[bidx], &val0[bidx], sizeof(FTYPE)*bseg);
		}
	}

	for(int i=0;i<nthreads;i++) 
		(*avg) += avg0[i];
	(*avg) /= (double)nr;
	free(avg0);

	for(int i=0;i<nr;i++) {
		int idx = (mcsr_cnt[i>>LOG_BH])*BH + (mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH])*((i&(BH-1))+1);
		int diff = row_ptr[i+1] - mcsr_e[idx-1]; 
		double r = ((double)diff - (*avg));
		(*vari) += r * r;

		if(diff >= STHRESHOLD) {
			int pp = (diff) / STHRESHOLD;
			for(int j=0; j<pp; j++) {
				special[*special_p] = i;
				special2[*special_p] = j * STHRESHOLD;
				(*special_p)++;
			}
		}
	}
	(*vari) /= (double)nr;

	gettimeofday(&endtime, NULL);

	// double elapsed0 = ((starttime.tv_sec-starttime0.tv_sec)*1000000 + starttime.tv_usec-starttime0.tv_usec)/1000000.0;
	// double p_elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
	// printf( "preprocessing %f,%f\n", elapsed0*1000, p_elapsed*1000);

	for(int i=0;i<nthreads;i++)
		free(bv[i]);
	for(int i=0;i<2;i++) {
		free(col_idx1[i]);
		free(coo[i]);
	}
	free(bv); free(col_idx1); free(coo);
}

void aspt_spmm_cpu(
	int * col_idx, FTYPE * val, 
	FTYPE * x, FTYPE * y, 
	int k, 
	int nr, double vari, 
	int *special, int *special2, int special_p, 
	int *mcsr_e, int *mcsr_cnt)
{
	// FTYPE *x, *y;
	// x = (FTYPE *)_mm_malloc(sizeof(FTYPE)*n*k,64);
	// y = (FTYPE *)_mm_malloc(sizeof(FTYPE)*nr*k,64);

	// row_ptr = (typeof(row_ptr))__builtin_assume_aligned(row_ptr, 64); // __assume_aligned(row_ptr, 64);
	// col_idx = (typeof(col_idx))__builtin_assume_aligned(col_idx, 64); // __assume_aligned(row_ptr, 64);
	// val = (typeof(val))__builtin_assume_aligned(val, 64); // __assume_aligned(val, 64);
	// x = (typeof(x))__builtin_assume_aligned(x, 64); // __assume_aligned(x, 64);
	// y = (typeof(y))__builtin_assume_aligned(y, 64); // __assume_aligned(y, 64);

	// #pragma omp parallel for
	// for(int i=0;i<n*k;i++) {
	// 	x[i] = 1;
	// }

	// memset(y, 0, sizeof(FTYPE)*nr*k);

	if(vari < 5000*1/1*1) {
		// for(int loop=0;loop<ITER;loop++)
		// {
			////begin
			// #pragma ivdep
			// #pragma vector aligned
			// #pragma temporal (x)
			#pragma omp parallel for schedule(dynamic, 1)
			for(int row_panel=0; row_panel<nr/BH; row_panel ++) {
				//dense
				int stride;
				for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

					for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
						int dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
						int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

						int interm = loc1 + (((loc2 - loc1)>>3)<<3);
						int j;
						for(j=loc1; j<interm; j+=8) {
							// #pragma ivdep
							// #pragma vector nontemporal (val)
							// #pragma prefetch x:_MM_HINT_T1
							// #pragma temporal (x)
							for(int kk=0; kk<k; kk++) {
								y[i*k+kk] = y[i*k+kk] + val[j] * x[col_idx[j]*k + kk]
									+ val[j+1] * x[col_idx[j+1]*k + kk]
									+ val[j+2] * x[col_idx[j+2]*k + kk]
									+ val[j+3] * x[col_idx[j+3]*k + kk]
									+ val[j+4] * x[col_idx[j+4]*k + kk]
									+ val[j+5] * x[col_idx[j+5]*k + kk]
									+ val[j+6] * x[col_idx[j+6]*k + kk]
									+ val[j+7] * x[col_idx[j+7]*k + kk];
							} 
						}
						for(; j<loc2; j++) {
							// #pragma ivdep
							// #pragma vector nontemporal (val)
							// #pragma prefetch y:_MM_HINT_T1
							// #pragma temporal (y)
							for(int kk=0; kk<k; kk++) {
								y[i*k + kk] += val[j] * x[col_idx[j]*k + kk];
							} 
						}
					}

				}
				//sparse
				for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {

					int dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					//printf("(%d %d %d %d %d)\n", i, row_ptr[i], loc1, row_ptr[i+1], loc2);
					//printf("%d %d %d %d %d %d %d\n", i, dummy, stride, row_ptr[i], loc1, row_ptr[i+1], loc2);


					int interm = loc1 + (((loc2 - loc1)>>3)<<3);
					int j;
					for(j=loc1; j<interm; j+=8) {
						// #pragma ivdep
						// #pragma vector nontemporal (val)
						// #pragma prefetch x:_MM_HINT_T1
						// #pragma temporal (x)
						for(int kk=0; kk<k; kk++) {
							y[i*k+kk] = y[i*k+kk] + val[j] * x[col_idx[j]*k + kk]
								+ val[j+1] * x[col_idx[j+1]*k + kk]
								+ val[j+2] * x[col_idx[j+2]*k + kk]
								+ val[j+3] * x[col_idx[j+3]*k + kk]
								+ val[j+4] * x[col_idx[j+4]*k + kk]
								+ val[j+5] * x[col_idx[j+5]*k + kk]
								+ val[j+6] * x[col_idx[j+6]*k + kk]
								+ val[j+7] * x[col_idx[j+7]*k + kk];
						} 
					}
					for(; j<loc2; j++) {
						// #pragma ivdep
						// #pragma vector nontemporal (val)
						// #pragma prefetch y:_MM_HINT_T1
						// #pragma temporal (y)
						for(int kk=0; kk<k; kk++) {
							y[i*k + kk] += val[j] * x[col_idx[j]*k + kk];
						} 
					}
				}
			}
			////end
		// }
	}
	else { // big var
		// for(int loop=0;loop<ITER;loop++)
		// {
			////begin
			// #pragma ivdep
			// #pragma vector aligned
			// #pragma temporal (x)
			#pragma omp parallel for schedule(dynamic, 1)
			for(int row_panel=0; row_panel<nr/BH; row_panel ++) {
				//dense
				int stride;
				for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

					for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
						int dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
						int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

						int interm = loc1 + (((loc2 - loc1)>>3)<<3);
						int j;
						for(j=loc1; j<interm; j+=8) {
							// #pragma ivdep
							// #pragma vector nontemporal (val)
							// #pragma prefetch x:_MM_HINT_T1
							// #pragma temporal (x)
							for(int kk=0; kk<k; kk++) {
								y[i*k+kk] = y[i*k+kk] + val[j] * x[col_idx[j]*k + kk]
									+ val[j+1] * x[col_idx[j+1]*k + kk]
									+ val[j+2] * x[col_idx[j+2]*k + kk]
									+ val[j+3] * x[col_idx[j+3]*k + kk]
									+ val[j+4] * x[col_idx[j+4]*k + kk]
									+ val[j+5] * x[col_idx[j+5]*k + kk]
									+ val[j+6] * x[col_idx[j+6]*k + kk]
									+ val[j+7] * x[col_idx[j+7]*k + kk];
							} 
						}
						for(; j<loc2; j++) {
							// #pragma ivdep
							// #pragma vector nontemporal (val)
							// #pragma prefetch y:_MM_HINT_T1
							// #pragma temporal (y)
							for(int kk=0; kk<k; kk++) {
								y[i*k + kk] += val[j] * x[col_idx[j]*k + kk];
							} 
						}
					}

				}
				//sparse
				for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {

					int dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					loc1 += ((loc2 - loc1)/STHRESHOLD)*STHRESHOLD;

					int interm = loc1 + (((loc2 - loc1)>>3)<<3);
					int j;
					for(j=loc1; j<interm; j+=8) {
						// #pragma ivdep
						// #pragma vector nontemporal (val)
						// #pragma prefetch x:_MM_HINT_T1
						// #pragma temporal (x)
						for(int kk=0; kk<k; kk++) {
							y[i*k+kk] = y[i*k+kk] + val[j] * x[col_idx[j]*k + kk]
								+ val[j+1] * x[col_idx[j+1]*k + kk]
								+ val[j+2] * x[col_idx[j+2]*k + kk]
								+ val[j+3] * x[col_idx[j+3]*k + kk]
								+ val[j+4] * x[col_idx[j+4]*k + kk]
								+ val[j+5] * x[col_idx[j+5]*k + kk]
								+ val[j+6] * x[col_idx[j+6]*k + kk]
								+ val[j+7] * x[col_idx[j+7]*k + kk];
						} 
					}
					for(; j<loc2; j++) {
						// #pragma ivdep
						// #pragma vector nontemporal (val)
						// #pragma prefetch y:_MM_HINT_T1
						// #pragma temporal (y)
						for(int kk=0; kk<k; kk++) {
							y[i*k + kk] += val[j] * x[col_idx[j]*k + kk];
						} 
					}
				}
			}
			// #pragma ivdep
			// #pragma vector aligned
			// #pragma temporal (x)
			#pragma omp parallel for schedule(dynamic, 1)
			for(int row_panel=0; row_panel<special_p;row_panel ++) {
				int i=special[row_panel];

				int dummy = mcsr_cnt[i>>LOG_BH]*BH + ((i&(BH-1))+1)*(mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH]);

				int loc1 = mcsr_e[dummy-1] + special2[row_panel];
				int loc2 = loc1 + STHRESHOLD;

				//int interm = loc1 + (((loc2 - loc1)>>3)<<3);
				int j;
				//assume to 128
				FTYPE temp_r[128]={0,};
				//for(int e=0;e<128;e++) {
				//	temp_r[e] = 0.0f;
				//}

				for(j=loc1; j<loc2; j+=8) {
					// #pragma ivdep
					// #pragma vector nontemporal (val)
					// #pragma prefetch x:_MM_HINT_T1
					// #pragma temporal (x)
					for(int kk=0; kk<k; kk++) {
						temp_r[kk] = temp_r[kk] + val[j] * x[col_idx[j]*k + kk]
							+ val[j+1] * x[col_idx[j+1]*k + kk]
							+ val[j+2] * x[col_idx[j+2]*k + kk]
							+ val[j+3] * x[col_idx[j+3]*k + kk]
							+ val[j+4] * x[col_idx[j+4]*k + kk]
							+ val[j+5] * x[col_idx[j+5]*k + kk]
							+ val[j+6] * x[col_idx[j+6]*k + kk]
							+ val[j+7] * x[col_idx[j+7]*k + kk];
					} 
				}
				// #pragma ivdep
				for(int kk=0; kk<k; kk++) {
					#pragma omp atomic
					y[i*k+kk] += temp_r[kk];
				}
			}
		// } // end loop
	}
}
