#include "aspt_sddmm_v2.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

/************************************************************************/
/* FUNCTION DEFINITIONS */

/************************** BB_EXCH.H (start)  **************************/
__device__ inline void exch_intxn(int &k0, int &v0, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k0, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v0, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    v0 = ex_v0;
}
// Exchange intersection for 2 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &v0, int &v1, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
}
// Exchange intersection for 4 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k2, int, v0, v2);
    if(bit) SWP(int, k1, k3, int, v1, v3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k2, int, v0, v2);
    if(bit) SWP(int, k1, k3, int, v1, v3);
}
// Exchange intersection for 8 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k6, int, v0, v6);
    if(bit) SWP(int, k1, k7, int, v1, v7);
    if(bit) SWP(int, k2, k4, int, v2, v4);
    if(bit) SWP(int, k3, k5, int, v3, v5);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k6, int, v0, v6);
    if(bit) SWP(int, k1, k7, int, v1, v7);
    if(bit) SWP(int, k2, k4, int, v2, v4);
    if(bit) SWP(int, k3, k5, int, v3, v5);
}
// Exchange intersection for 16 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k14, int, v0, v14);
    if(bit) SWP(int, k1, k15, int, v1, v15);
    if(bit) SWP(int, k2, k12, int, v2, v12);
    if(bit) SWP(int, k3, k13, int, v3, v13);
    if(bit) SWP(int, k4, k10, int, v4, v10);
    if(bit) SWP(int, k5, k11, int, v5, v11);
    if(bit) SWP(int, k6, k8, int, v6, v8);
    if(bit) SWP(int, k7, k9, int, v7, v9);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor(k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor(v9, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor(ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor(ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor(k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor(v11, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor(ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor(ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor(k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor(v13, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor(ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor(ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor(k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor(v15, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor(ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k14, int, v0, v14);
    if(bit) SWP(int, k1, k15, int, v1, v15);
    if(bit) SWP(int, k2, k12, int, v2, v12);
    if(bit) SWP(int, k3, k13, int, v3, v13);
    if(bit) SWP(int, k4, k10, int, v4, v10);
    if(bit) SWP(int, k5, k11, int, v5, v11);
    if(bit) SWP(int, k6, k8, int, v6, v8);
    if(bit) SWP(int, k7, k9, int, v7, v9);
}
// Exchange intersection for 32 keys.
__device__ inline void exch_intxn(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &k16, int &k17, int &k18, int &k19, int &k20, int &k21, int &k22, int &k23, int &k24, int &k25, int &k26, int &k27, int &k28, int &k29, int &k30, int &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k30, int, v0, v30);
    if(bit) SWP(int, k1, k31, int, v1, v31);
    if(bit) SWP(int, k2, k28, int, v2, v28);
    if(bit) SWP(int, k3, k29, int, v3, v29);
    if(bit) SWP(int, k4, k26, int, v4, v26);
    if(bit) SWP(int, k5, k27, int, v5, v27);
    if(bit) SWP(int, k6, k24, int, v6, v24);
    if(bit) SWP(int, k7, k25, int, v7, v25);
    if(bit) SWP(int, k8, k22, int, v8, v22);
    if(bit) SWP(int, k9, k23, int, v9, v23);
    if(bit) SWP(int, k10, k20, int, v10, v20);
    if(bit) SWP(int, k11, k21, int, v11, v21);
    if(bit) SWP(int, k12, k18, int, v12, v18);
    if(bit) SWP(int, k13, k19, int, v13, v19);
    if(bit) SWP(int, k14, k16, int, v14, v16);
    if(bit) SWP(int, k15, k17, int, v15, v17);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor(k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor(v9, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor(ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor(ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor(k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor(v11, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor(ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor(ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor(k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor(v13, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor(ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor(ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor(k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor(v15, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor(ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor(ex_v1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor(k17, mask);
    ex_v0 = v16;
    ex_v1 = __shfl_xor(v17, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k16 = ex_k0;
    k17 = __shfl_xor(ex_k1, mask);
    v16 = ex_v0;
    v17 = __shfl_xor(ex_v1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor(k19, mask);
    ex_v0 = v18;
    ex_v1 = __shfl_xor(v19, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k18 = ex_k0;
    k19 = __shfl_xor(ex_k1, mask);
    v18 = ex_v0;
    v19 = __shfl_xor(ex_v1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor(k21, mask);
    ex_v0 = v20;
    ex_v1 = __shfl_xor(v21, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k20 = ex_k0;
    k21 = __shfl_xor(ex_k1, mask);
    v20 = ex_v0;
    v21 = __shfl_xor(ex_v1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor(k23, mask);
    ex_v0 = v22;
    ex_v1 = __shfl_xor(v23, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k22 = ex_k0;
    k23 = __shfl_xor(ex_k1, mask);
    v22 = ex_v0;
    v23 = __shfl_xor(ex_v1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor(k25, mask);
    ex_v0 = v24;
    ex_v1 = __shfl_xor(v25, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k24 = ex_k0;
    k25 = __shfl_xor(ex_k1, mask);
    v24 = ex_v0;
    v25 = __shfl_xor(ex_v1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor(k27, mask);
    ex_v0 = v26;
    ex_v1 = __shfl_xor(v27, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k26 = ex_k0;
    k27 = __shfl_xor(ex_k1, mask);
    v26 = ex_v0;
    v27 = __shfl_xor(ex_v1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor(k29, mask);
    ex_v0 = v28;
    ex_v1 = __shfl_xor(v29, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k28 = ex_k0;
    k29 = __shfl_xor(ex_k1, mask);
    v28 = ex_v0;
    v29 = __shfl_xor(ex_v1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor(k31, mask);
    ex_v0 = v30;
    ex_v1 = __shfl_xor(v31, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k30 = ex_k0;
    k31 = __shfl_xor(ex_k1, mask);
    v30 = ex_v0;
    v31 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k30, int, v0, v30);
    if(bit) SWP(int, k1, k31, int, v1, v31);
    if(bit) SWP(int, k2, k28, int, v2, v28);
    if(bit) SWP(int, k3, k29, int, v3, v29);
    if(bit) SWP(int, k4, k26, int, v4, v26);
    if(bit) SWP(int, k5, k27, int, v5, v27);
    if(bit) SWP(int, k6, k24, int, v6, v24);
    if(bit) SWP(int, k7, k25, int, v7, v25);
    if(bit) SWP(int, k8, k22, int, v8, v22);
    if(bit) SWP(int, k9, k23, int, v9, v23);
    if(bit) SWP(int, k10, k20, int, v10, v20);
    if(bit) SWP(int, k11, k21, int, v11, v21);
    if(bit) SWP(int, k12, k18, int, v12, v18);
    if(bit) SWP(int, k13, k19, int, v13, v19);
    if(bit) SWP(int, k14, k16, int, v14, v16);
    if(bit) SWP(int, k15, k17, int, v15, v17);
}
// Exchange parallel for 1 keys.
__device__ inline void exch_paral(int &k0, int &v0, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k0, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v0, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    v0 = ex_v0;
}
// Exchange parallel for 2 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &v0, int &v1, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k1, int, v0, v1);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k1, int, v0, v1);
}
// Exchange parallel for 4 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &v0, int &v1, int &v2, int &v3, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
}
// Exchange parallel for 8 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
}
// Exchange parallel for 16 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
    if(bit) SWP(int, k8, k9, int, v8, v9);
    if(bit) SWP(int, k10, k11, int, v10, v11);
    if(bit) SWP(int, k12, k13, int, v12, v13);
    if(bit) SWP(int, k14, k15, int, v14, v15);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor(k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor(v9, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor(ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor(ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor(k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor(v11, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor(ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor(ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor(k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor(v13, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor(ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor(ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor(k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor(v15, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor(ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
    if(bit) SWP(int, k8, k9, int, v8, v9);
    if(bit) SWP(int, k10, k11, int, v10, v11);
    if(bit) SWP(int, k12, k13, int, v12, v13);
    if(bit) SWP(int, k14, k15, int, v14, v15);
}
// Exchange parallel for 32 keys.
__device__ inline void exch_paral(int &k0, int &k1, int &k2, int &k3, int &k4, int &k5, int &k6, int &k7, int &k8, int &k9, int &k10, int &k11, int &k12, int &k13, int &k14, int &k15, int &k16, int &k17, int &k18, int &k19, int &k20, int &k21, int &k22, int &k23, int &k24, int &k25, int &k26, int &k27, int &k28, int &k29, int &k30, int &k31, int &v0, int &v1, int &v2, int &v3, int &v4, int &v5, int &v6, int &v7, int &v8, int &v9, int &v10, int &v11, int &v12, int &v13, int &v14, int &v15, int &v16, int &v17, int &v18, int &v19, int &v20, int &v21, int &v22, int &v23, int &v24, int &v25, int &v26, int &v27, int &v28, int &v29, int &v30, int &v31, int mask, const int bit) {
    int ex_k0, ex_k1;
    int ex_v0, ex_v1;
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
    if(bit) SWP(int, k8, k9, int, v8, v9);
    if(bit) SWP(int, k10, k11, int, v10, v11);
    if(bit) SWP(int, k12, k13, int, v12, v13);
    if(bit) SWP(int, k14, k15, int, v14, v15);
    if(bit) SWP(int, k16, k17, int, v16, v17);
    if(bit) SWP(int, k18, k19, int, v18, v19);
    if(bit) SWP(int, k20, k21, int, v20, v21);
    if(bit) SWP(int, k22, k23, int, v22, v23);
    if(bit) SWP(int, k24, k25, int, v24, v25);
    if(bit) SWP(int, k26, k27, int, v26, v27);
    if(bit) SWP(int, k28, k29, int, v28, v29);
    if(bit) SWP(int, k30, k31, int, v30, v31);
    ex_k0 = k0;
    ex_k1 = __shfl_xor(k1, mask);
    ex_v0 = v0;
    ex_v1 = __shfl_xor(v1, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k0 = ex_k0;
    k1 = __shfl_xor(ex_k1, mask);
    v0 = ex_v0;
    v1 = __shfl_xor(ex_v1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor(k3, mask);
    ex_v0 = v2;
    ex_v1 = __shfl_xor(v3, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k2 = ex_k0;
    k3 = __shfl_xor(ex_k1, mask);
    v2 = ex_v0;
    v3 = __shfl_xor(ex_v1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor(k5, mask);
    ex_v0 = v4;
    ex_v1 = __shfl_xor(v5, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k4 = ex_k0;
    k5 = __shfl_xor(ex_k1, mask);
    v4 = ex_v0;
    v5 = __shfl_xor(ex_v1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor(k7, mask);
    ex_v0 = v6;
    ex_v1 = __shfl_xor(v7, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k6 = ex_k0;
    k7 = __shfl_xor(ex_k1, mask);
    v6 = ex_v0;
    v7 = __shfl_xor(ex_v1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor(k9, mask);
    ex_v0 = v8;
    ex_v1 = __shfl_xor(v9, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k8 = ex_k0;
    k9 = __shfl_xor(ex_k1, mask);
    v8 = ex_v0;
    v9 = __shfl_xor(ex_v1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor(k11, mask);
    ex_v0 = v10;
    ex_v1 = __shfl_xor(v11, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k10 = ex_k0;
    k11 = __shfl_xor(ex_k1, mask);
    v10 = ex_v0;
    v11 = __shfl_xor(ex_v1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor(k13, mask);
    ex_v0 = v12;
    ex_v1 = __shfl_xor(v13, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k12 = ex_k0;
    k13 = __shfl_xor(ex_k1, mask);
    v12 = ex_v0;
    v13 = __shfl_xor(ex_v1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor(k15, mask);
    ex_v0 = v14;
    ex_v1 = __shfl_xor(v15, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k14 = ex_k0;
    k15 = __shfl_xor(ex_k1, mask);
    v14 = ex_v0;
    v15 = __shfl_xor(ex_v1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor(k17, mask);
    ex_v0 = v16;
    ex_v1 = __shfl_xor(v17, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k16 = ex_k0;
    k17 = __shfl_xor(ex_k1, mask);
    v16 = ex_v0;
    v17 = __shfl_xor(ex_v1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor(k19, mask);
    ex_v0 = v18;
    ex_v1 = __shfl_xor(v19, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k18 = ex_k0;
    k19 = __shfl_xor(ex_k1, mask);
    v18 = ex_v0;
    v19 = __shfl_xor(ex_v1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor(k21, mask);
    ex_v0 = v20;
    ex_v1 = __shfl_xor(v21, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k20 = ex_k0;
    k21 = __shfl_xor(ex_k1, mask);
    v20 = ex_v0;
    v21 = __shfl_xor(ex_v1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor(k23, mask);
    ex_v0 = v22;
    ex_v1 = __shfl_xor(v23, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k22 = ex_k0;
    k23 = __shfl_xor(ex_k1, mask);
    v22 = ex_v0;
    v23 = __shfl_xor(ex_v1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor(k25, mask);
    ex_v0 = v24;
    ex_v1 = __shfl_xor(v25, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k24 = ex_k0;
    k25 = __shfl_xor(ex_k1, mask);
    v24 = ex_v0;
    v25 = __shfl_xor(ex_v1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor(k27, mask);
    ex_v0 = v26;
    ex_v1 = __shfl_xor(v27, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k26 = ex_k0;
    k27 = __shfl_xor(ex_k1, mask);
    v26 = ex_v0;
    v27 = __shfl_xor(ex_v1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor(k29, mask);
    ex_v0 = v28;
    ex_v1 = __shfl_xor(v29, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k28 = ex_k0;
    k29 = __shfl_xor(ex_k1, mask);
    v28 = ex_v0;
    v29 = __shfl_xor(ex_v1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor(k31, mask);
    ex_v0 = v30;
    ex_v1 = __shfl_xor(v31, mask);
    CMP_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    if(bit) EQL_SWP(int, ex_k0, ex_k1, int, ex_v0, ex_v1);
    k30 = ex_k0;
    k31 = __shfl_xor(ex_k1, mask);
    v30 = ex_v0;
    v31 = __shfl_xor(ex_v1, mask);
    if(bit) SWP(int, k0, k1, int, v0, v1);
    if(bit) SWP(int, k2, k3, int, v2, v3);
    if(bit) SWP(int, k4, k5, int, v4, v5);
    if(bit) SWP(int, k6, k7, int, v6, v7);
    if(bit) SWP(int, k8, k9, int, v8, v9);
    if(bit) SWP(int, k10, k11, int, v10, v11);
    if(bit) SWP(int, k12, k13, int, v12, v13);
    if(bit) SWP(int, k14, k15, int, v14, v15);
    if(bit) SWP(int, k16, k17, int, v16, v17);
    if(bit) SWP(int, k18, k19, int, v18, v19);
    if(bit) SWP(int, k20, k21, int, v20, v21);
    if(bit) SWP(int, k22, k23, int, v22, v23);
    if(bit) SWP(int, k24, k25, int, v24, v25);
    if(bit) SWP(int, k26, k27, int, v26, v27);
    if(bit) SWP(int, k28, k29, int, v28, v29);
    if(bit) SWP(int, k30, k31, int, v30, v31);
}
/*************************** BB_EXCH.H (end)  ***************************/

/************************ BB_COMPUT_S.H (start)  ************************/
__device__ int find_kth3(int* a, int aCount, int* b, int bCount, int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);
 
    while(begin < end) {
        int mid = (begin + end)>> 1;
        int aKey = a[mid];
        int bKey = b[diag - 1 - mid];
        bool pred = aKey <= bKey;
        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

__global__ void gen_copy(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = gid;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        if(seg_size == 1)
        {
            keyB[k] = key[k];
            valB[k] = val[k];
        }
    }
}

/* block tcf subwarp coalesced quiet real_kern */
/*   256   1       2     false  true      true */
__global__ void gen_bk256_wp2_tc1_r2_r2_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    int rg_k0 ;
    int rg_v0 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        // sort 2 elements
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,
                   rg_v0 ,
                   0x1,bit1);
        if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0 ] = rg_k0 ;
        if((tid<<0)+0 <seg_size) valB[k+(tid<<0)+0 ] = val[k+rg_v0 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2       2     false  true      true */
__global__ void gen_bk128_wp2_tc2_r3_r4_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_v0 ;
    int rg_v1 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+2   <seg_size) rg_v1  = tid+2   ;
        // sort 4 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,
                   rg_v0 ,rg_v1 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
        if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       2     false  true      true */
__global__ void gen_bk128_wp2_tc4_r5_r8_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:INT_MAX;
        rg_k2  = (tid+4   <seg_size)?key[k+tid+4   ]:INT_MAX;
        rg_k3  = (tid+6   <seg_size)?key[k+tid+6   ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+2   <seg_size) rg_v1  = tid+2   ;
        if(tid+4   <seg_size) rg_v2  = tid+4   ;
        if(tid+6   <seg_size) rg_v3  = tid+6   ;
        // sort 8 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       4      true  true      true */
__global__ void gen_bk128_wp4_tc4_r9_r16_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>2);
    const int tid = (threadIdx.x & 3);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/8)*8;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+4   <seg_size)?key[k+tid+4   ]:INT_MAX;
        rg_k2  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k3  = (tid+12  <seg_size)?key[k+tid+12  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+4   <seg_size) rg_v1  = tid+4   ;
        if(tid+8   <seg_size) rg_v2  = tid+8   ;
        if(tid+12  <seg_size) rg_v3  = tid+12  ;
        // sort 16 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        int base = (lane_id/16)*16;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0 ];
        kk = __shfl(k, 4 );
        ss = __shfl(seg_size, 4 );
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v0 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2 ];
        kk = __shfl(k, 12);
        ss = __shfl(seg_size, 12);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1 ];
        kk = __shfl(k, 20);
        ss = __shfl(seg_size, 20);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v1 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if((lane_id>>4)==0&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3 ];
        kk = __shfl(k, 28);
        ss = __shfl(seg_size, 28);
        if((lane_id>>4)==1&lane_id-base<ss) valB[kk+lane_id-base] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       8      true  true      true */
__global__ void gen_bk128_wp8_tc4_r17_r32_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+8   <seg_size) rg_v1  = tid+8   ;
        if(tid+16  <seg_size) rg_v2  = tid+16  ;
        if(tid+24  <seg_size) rg_v3  = tid+24  ;
        // sort 32 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4      16      true  true      true */
__global__ void gen_bk128_wp16_tc4_r33_r64_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>4);
    const int tid = (threadIdx.x & 15);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int normalized_bin_size = (bin_size/2)*2;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k2  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k3  = (tid+48  <seg_size)?key[k+tid+48  ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+16  <seg_size) rg_v1  = tid+16  ;
        if(tid+32  <seg_size) rg_v2  = tid+32  ;
        if(tid+48  <seg_size) rg_v3  = tid+48  ;
        // sort 64 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        if(lane_id&0x1 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x4 );
        rg_k3  = __shfl_xor(rg_k3 , 0x4 );
        rg_v1  = __shfl_xor(rg_v1 , 0x4 );
        rg_v3  = __shfl_xor(rg_v3 , 0x4 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x8 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        rg_k2  = __shfl_xor(rg_k2 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_v2  = __shfl_xor(rg_v2 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
        if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = val[k+rg_v0 ];
        if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = val[k+rg_v1 ];
        if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = val[k+rg_v2 ];
        if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = val[k+rg_v3 ];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256  16       8      true  true      true */
__global__ void gen_bk256_wp8_tc16_r65_r128_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_k4 ;
    int rg_k5 ;
    int rg_k6 ;
    int rg_k7 ;
    int rg_k8 ;
    int rg_k9 ;
    int rg_k10;
    int rg_k11;
    int rg_k12;
    int rg_k13;
    int rg_k14;
    int rg_k15;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int rg_v4 ;
    int rg_v5 ;
    int rg_v6 ;
    int rg_v7 ;
    int rg_v8 ;
    int rg_v9 ;
    int rg_v10;
    int rg_v11;
    int rg_v12;
    int rg_v13;
    int rg_v14;
    int rg_v15;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:INT_MAX;
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:INT_MAX;
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:INT_MAX;
        rg_k4  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k5  = (tid+40  <seg_size)?key[k+tid+40  ]:INT_MAX;
        rg_k6  = (tid+48  <seg_size)?key[k+tid+48  ]:INT_MAX;
        rg_k7  = (tid+56  <seg_size)?key[k+tid+56  ]:INT_MAX;
        rg_k8  = (tid+64  <seg_size)?key[k+tid+64  ]:INT_MAX;
        rg_k9  = (tid+72  <seg_size)?key[k+tid+72  ]:INT_MAX;
        rg_k10 = (tid+80  <seg_size)?key[k+tid+80  ]:INT_MAX;
        rg_k11 = (tid+88  <seg_size)?key[k+tid+88  ]:INT_MAX;
        rg_k12 = (tid+96  <seg_size)?key[k+tid+96  ]:INT_MAX;
        rg_k13 = (tid+104 <seg_size)?key[k+tid+104 ]:INT_MAX;
        rg_k14 = (tid+112 <seg_size)?key[k+tid+112 ]:INT_MAX;
        rg_k15 = (tid+120 <seg_size)?key[k+tid+120 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+8   <seg_size) rg_v1  = tid+8   ;
        if(tid+16  <seg_size) rg_v2  = tid+16  ;
        if(tid+24  <seg_size) rg_v3  = tid+24  ;
        if(tid+32  <seg_size) rg_v4  = tid+32  ;
        if(tid+40  <seg_size) rg_v5  = tid+40  ;
        if(tid+48  <seg_size) rg_v6  = tid+48  ;
        if(tid+56  <seg_size) rg_v7  = tid+56  ;
        if(tid+64  <seg_size) rg_v8  = tid+64  ;
        if(tid+72  <seg_size) rg_v9  = tid+72  ;
        if(tid+80  <seg_size) rg_v10 = tid+80  ;
        if(tid+88  <seg_size) rg_v11 = tid+88  ;
        if(tid+96  <seg_size) rg_v12 = tid+96  ;
        if(tid+104 <seg_size) rg_v13 = tid+104 ;
        if(tid+112 <seg_size) rg_v14 = tid+112 ;
        if(tid+120 <seg_size) rg_v15 = tid+120 ;
        // sort 128 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        CMP_SWP(int,rg_k4 ,rg_k7 ,int,rg_v4 ,rg_v7 );
        CMP_SWP(int,rg_k5 ,rg_k6 ,int,rg_v5 ,rg_v6 );
        CMP_SWP(int,rg_k8 ,rg_k11,int,rg_v8 ,rg_v11);
        CMP_SWP(int,rg_k9 ,rg_k10,int,rg_v9 ,rg_v10);
        CMP_SWP(int,rg_k12,rg_k15,int,rg_v12,rg_v15);
        CMP_SWP(int,rg_k13,rg_k14,int,rg_v13,rg_v14);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k7 ,int,rg_v0 ,rg_v7 );
        CMP_SWP(int,rg_k1 ,rg_k6 ,int,rg_v1 ,rg_v6 );
        CMP_SWP(int,rg_k2 ,rg_k5 ,int,rg_v2 ,rg_v5 );
        CMP_SWP(int,rg_k3 ,rg_k4 ,int,rg_v3 ,rg_v4 );
        CMP_SWP(int,rg_k8 ,rg_k15,int,rg_v8 ,rg_v15);
        CMP_SWP(int,rg_k9 ,rg_k14,int,rg_v9 ,rg_v14);
        CMP_SWP(int,rg_k10,rg_k13,int,rg_v10,rg_v13);
        CMP_SWP(int,rg_k11,rg_k12,int,rg_v11,rg_v12);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(int,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(int,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(int,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k15,int,rg_v0 ,rg_v15);
        CMP_SWP(int,rg_k1 ,rg_k14,int,rg_v1 ,rg_v14);
        CMP_SWP(int,rg_k2 ,rg_k13,int,rg_v2 ,rg_v13);
        CMP_SWP(int,rg_k3 ,rg_k12,int,rg_v3 ,rg_v12);
        CMP_SWP(int,rg_k4 ,rg_k11,int,rg_v4 ,rg_v11);
        CMP_SWP(int,rg_k5 ,rg_k10,int,rg_v5 ,rg_v10);
        CMP_SWP(int,rg_k6 ,rg_k9 ,int,rg_v6 ,rg_v9 );
        CMP_SWP(int,rg_k7 ,rg_k8 ,int,rg_v7 ,rg_v8 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(int,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(int,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(int,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(int,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(int,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(int,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(int,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(int,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(int,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(int,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(int,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(int,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(int,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(int,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(int,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(int,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(int,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(int,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(int,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(int,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(int,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(int,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(int,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(int,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(int,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(int,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(int,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(int,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(int,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(int,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(int,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(int,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   rg_v8 ,rg_v9 ,rg_v10,rg_v11,rg_v12,rg_v13,rg_v14,rg_v15,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k8 ,int,rg_v0 ,rg_v8 );
        CMP_SWP(int,rg_k1 ,rg_k9 ,int,rg_v1 ,rg_v9 );
        CMP_SWP(int,rg_k2 ,rg_k10,int,rg_v2 ,rg_v10);
        CMP_SWP(int,rg_k3 ,rg_k11,int,rg_v3 ,rg_v11);
        CMP_SWP(int,rg_k4 ,rg_k12,int,rg_v4 ,rg_v12);
        CMP_SWP(int,rg_k5 ,rg_k13,int,rg_v5 ,rg_v13);
        CMP_SWP(int,rg_k6 ,rg_k14,int,rg_v6 ,rg_v14);
        CMP_SWP(int,rg_k7 ,rg_k15,int,rg_v7 ,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k12,int,rg_v8 ,rg_v12);
        CMP_SWP(int,rg_k9 ,rg_k13,int,rg_v9 ,rg_v13);
        CMP_SWP(int,rg_k10,rg_k14,int,rg_v10,rg_v14);
        CMP_SWP(int,rg_k11,rg_k15,int,rg_v11,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k10,int,rg_v8 ,rg_v10);
        CMP_SWP(int,rg_k9 ,rg_k11,int,rg_v9 ,rg_v11);
        CMP_SWP(int,rg_k12,rg_k14,int,rg_v12,rg_v14);
        CMP_SWP(int,rg_k13,rg_k15,int,rg_v13,rg_v15);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        CMP_SWP(int,rg_k8 ,rg_k9 ,int,rg_v8 ,rg_v9 );
        CMP_SWP(int,rg_k10,rg_k11,int,rg_v10,rg_v11);
        CMP_SWP(int,rg_k12,rg_k13,int,rg_v12,rg_v13);
        CMP_SWP(int,rg_k14,rg_k15,int,rg_v14,rg_v15);
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        if(lane_id&0x1 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x1 ) SWP(int, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x1 ) SWP(int, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x1 ) SWP(int, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x1 ) SWP(int, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x1 ) SWP(int, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x1 ) SWP(int, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(int, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(int, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        if(lane_id&0x2 ) SWP(int, rg_k8 , rg_k10, int, rg_v8 , rg_v10);
        if(lane_id&0x2 ) SWP(int, rg_k9 , rg_k11, int, rg_v9 , rg_v11);
        if(lane_id&0x2 ) SWP(int, rg_k12, rg_k14, int, rg_v12, rg_v14);
        if(lane_id&0x2 ) SWP(int, rg_k13, rg_k15, int, rg_v13, rg_v15);
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(int, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(int, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        if(lane_id&0x4 ) SWP(int, rg_k8 , rg_k12, int, rg_v8 , rg_v12);
        if(lane_id&0x4 ) SWP(int, rg_k9 , rg_k13, int, rg_v9 , rg_v13);
        if(lane_id&0x4 ) SWP(int, rg_k10, rg_k14, int, rg_v10, rg_v14);
        if(lane_id&0x4 ) SWP(int, rg_k11, rg_k15, int, rg_v11, rg_v15);
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
        if(lane_id&0x8 ) SWP(int, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
        if(lane_id&0x8 ) SWP(int, rg_k2 , rg_k10, int, rg_v2 , rg_v10);
        if(lane_id&0x8 ) SWP(int, rg_k3 , rg_k11, int, rg_v3 , rg_v11);
        if(lane_id&0x8 ) SWP(int, rg_k4 , rg_k12, int, rg_v4 , rg_v12);
        if(lane_id&0x8 ) SWP(int, rg_k5 , rg_k13, int, rg_v5 , rg_v13);
        if(lane_id&0x8 ) SWP(int, rg_k6 , rg_k14, int, rg_v6 , rg_v14);
        if(lane_id&0x8 ) SWP(int, rg_k7 , rg_k15, int, rg_v7 , rg_v15);
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x10) SWP(int, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x10) SWP(int, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x10) SWP(int, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x10) SWP(int, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x10) SWP(int, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x10) SWP(int, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k4 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k6 ;
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k8 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k10;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k12;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k14;
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k5 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k7 ;
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k9 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k11;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k13;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k15;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v2 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v4 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v6 ];
        kk = __shfl(k, 8 );
        ss = __shfl(seg_size, 8 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v8 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v10];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v12];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v14];
        kk = __shfl(k, 16);
        ss = __shfl(seg_size, 16);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v1 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v3 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v5 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v7 ];
        kk = __shfl(k, 24);
        ss = __shfl(seg_size, 24);
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v9 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v11];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v13];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v15];
    } else if(bin_it < bin_size) {
        if((tid<<4)+0 <seg_size) keyB[k+(tid<<4)+0 ] = rg_k0 ;
        if((tid<<4)+1 <seg_size) keyB[k+(tid<<4)+1 ] = rg_k1 ;
        if((tid<<4)+2 <seg_size) keyB[k+(tid<<4)+2 ] = rg_k2 ;
        if((tid<<4)+3 <seg_size) keyB[k+(tid<<4)+3 ] = rg_k3 ;
        if((tid<<4)+4 <seg_size) keyB[k+(tid<<4)+4 ] = rg_k4 ;
        if((tid<<4)+5 <seg_size) keyB[k+(tid<<4)+5 ] = rg_k5 ;
        if((tid<<4)+6 <seg_size) keyB[k+(tid<<4)+6 ] = rg_k6 ;
        if((tid<<4)+7 <seg_size) keyB[k+(tid<<4)+7 ] = rg_k7 ;
        if((tid<<4)+8 <seg_size) keyB[k+(tid<<4)+8 ] = rg_k8 ;
        if((tid<<4)+9 <seg_size) keyB[k+(tid<<4)+9 ] = rg_k9 ;
        if((tid<<4)+10<seg_size) keyB[k+(tid<<4)+10] = rg_k10;
        if((tid<<4)+11<seg_size) keyB[k+(tid<<4)+11] = rg_k11;
        if((tid<<4)+12<seg_size) keyB[k+(tid<<4)+12] = rg_k12;
        if((tid<<4)+13<seg_size) keyB[k+(tid<<4)+13] = rg_k13;
        if((tid<<4)+14<seg_size) keyB[k+(tid<<4)+14] = rg_k14;
        if((tid<<4)+15<seg_size) keyB[k+(tid<<4)+15] = rg_k15;
        if((tid<<4)+0 <seg_size) valB[k+(tid<<4)+0 ] = val[k+rg_v0 ];
        if((tid<<4)+1 <seg_size) valB[k+(tid<<4)+1 ] = val[k+rg_v1 ];
        if((tid<<4)+2 <seg_size) valB[k+(tid<<4)+2 ] = val[k+rg_v2 ];
        if((tid<<4)+3 <seg_size) valB[k+(tid<<4)+3 ] = val[k+rg_v3 ];
        if((tid<<4)+4 <seg_size) valB[k+(tid<<4)+4 ] = val[k+rg_v4 ];
        if((tid<<4)+5 <seg_size) valB[k+(tid<<4)+5 ] = val[k+rg_v5 ];
        if((tid<<4)+6 <seg_size) valB[k+(tid<<4)+6 ] = val[k+rg_v6 ];
        if((tid<<4)+7 <seg_size) valB[k+(tid<<4)+7 ] = val[k+rg_v7 ];
        if((tid<<4)+8 <seg_size) valB[k+(tid<<4)+8 ] = val[k+rg_v8 ];
        if((tid<<4)+9 <seg_size) valB[k+(tid<<4)+9 ] = val[k+rg_v9 ];
        if((tid<<4)+10<seg_size) valB[k+(tid<<4)+10] = val[k+rg_v10];
        if((tid<<4)+11<seg_size) valB[k+(tid<<4)+11] = val[k+rg_v11];
        if((tid<<4)+12<seg_size) valB[k+(tid<<4)+12] = val[k+rg_v12];
        if((tid<<4)+13<seg_size) valB[k+(tid<<4)+13] = val[k+rg_v13];
        if((tid<<4)+14<seg_size) valB[k+(tid<<4)+14] = val[k+rg_v14];
        if((tid<<4)+15<seg_size) valB[k+(tid<<4)+15] = val[k+rg_v15];
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256   8      32      true  true      true */
__global__ void gen_bk256_wp32_tc8_r129_r256_strd(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>5);
    const int tid = (threadIdx.x & 31);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_k4 ;
    int rg_k5 ;
    int rg_k6 ;
    int rg_k7 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int rg_v4 ;
    int rg_v5 ;
    int rg_v6 ;
    int rg_v7 ;
    int normalized_bin_size = (bin_size/1)*1;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:INT_MAX;
        rg_k1  = (tid+32  <seg_size)?key[k+tid+32  ]:INT_MAX;
        rg_k2  = (tid+64  <seg_size)?key[k+tid+64  ]:INT_MAX;
        rg_k3  = (tid+96  <seg_size)?key[k+tid+96  ]:INT_MAX;
        rg_k4  = (tid+128 <seg_size)?key[k+tid+128 ]:INT_MAX;
        rg_k5  = (tid+160 <seg_size)?key[k+tid+160 ]:INT_MAX;
        rg_k6  = (tid+192 <seg_size)?key[k+tid+192 ]:INT_MAX;
        rg_k7  = (tid+224 <seg_size)?key[k+tid+224 ]:INT_MAX;
        if(tid+0   <seg_size) rg_v0  = tid+0   ;
        if(tid+32  <seg_size) rg_v1  = tid+32  ;
        if(tid+64  <seg_size) rg_v2  = tid+64  ;
        if(tid+96  <seg_size) rg_v3  = tid+96  ;
        if(tid+128 <seg_size) rg_v4  = tid+128 ;
        if(tid+160 <seg_size) rg_v5  = tid+160 ;
        if(tid+192 <seg_size) rg_v6  = tid+192 ;
        if(tid+224 <seg_size) rg_v7  = tid+224 ;
        // sort 256 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
        CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
        CMP_SWP(int,rg_k4 ,rg_k7 ,int,rg_v4 ,rg_v7 );
        CMP_SWP(int,rg_k5 ,rg_k6 ,int,rg_v5 ,rg_v6 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k7 ,int,rg_v0 ,rg_v7 );
        CMP_SWP(int,rg_k1 ,rg_k6 ,int,rg_v1 ,rg_v6 );
        CMP_SWP(int,rg_k2 ,rg_k5 ,int,rg_v2 ,rg_v5 );
        CMP_SWP(int,rg_k3 ,rg_k4 ,int,rg_v3 ,rg_v4 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x3,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x7,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0xf,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
        // exch_intxn: generate exch_intxn()
        exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1f,bit5);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x8,bit4);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x4,bit3);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x2,bit2);
        // exch_paral: generate exch_paral()
        exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                   rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,rg_v4 ,rg_v5 ,rg_v6 ,rg_v7 ,
                   0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k4 ,int,rg_v0 ,rg_v4 );
        CMP_SWP(int,rg_k1 ,rg_k5 ,int,rg_v1 ,rg_v5 );
        CMP_SWP(int,rg_k2 ,rg_k6 ,int,rg_v2 ,rg_v6 );
        CMP_SWP(int,rg_k3 ,rg_k7 ,int,rg_v3 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
        CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k6 ,int,rg_v4 ,rg_v6 );
        CMP_SWP(int,rg_k5 ,rg_k7 ,int,rg_v5 ,rg_v7 );
        // exch_paral: switch to exch_local()
        CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        CMP_SWP(int,rg_k4 ,rg_k5 ,int,rg_v4 ,rg_v5 );
        CMP_SWP(int,rg_k6 ,rg_k7 ,int,rg_v6 ,rg_v7 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        if(lane_id&0x1 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x1 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x1 ) SWP(int, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x1 ) SWP(int, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(int, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(int, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(int, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(int, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_k1  = __shfl_xor(rg_k1 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_k5  = __shfl_xor(rg_k5 , 0x8 );
        rg_k7  = __shfl_xor(rg_k7 , 0x8 );
        rg_v1  = __shfl_xor(rg_v1 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_v5  = __shfl_xor(rg_v5 , 0x8 );
        rg_v7  = __shfl_xor(rg_v7 , 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x8 ) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x8 ) SWP(int, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x8 ) SWP(int, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        rg_k1  = __shfl_xor(rg_k1 , 0x8 );
        rg_k3  = __shfl_xor(rg_k3 , 0x8 );
        rg_k5  = __shfl_xor(rg_k5 , 0x8 );
        rg_k7  = __shfl_xor(rg_k7 , 0x8 );
        rg_v1  = __shfl_xor(rg_v1 , 0x8 );
        rg_v3  = __shfl_xor(rg_v3 , 0x8 );
        rg_v5  = __shfl_xor(rg_v5 , 0x8 );
        rg_v7  = __shfl_xor(rg_v7 , 0x8 );
        rg_k2  = __shfl_xor(rg_k2 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k6  = __shfl_xor(rg_k6 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_v2  = __shfl_xor(rg_v2 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v6  = __shfl_xor(rg_v6 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x10) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x10) SWP(int, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x10) SWP(int, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        rg_k2  = __shfl_xor(rg_k2 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k6  = __shfl_xor(rg_k6 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_v2  = __shfl_xor(rg_v2 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v6  = __shfl_xor(rg_v6 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        int kk;
        int ss;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k4 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k1 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k5 ;
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k2 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k6 ;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k3 ;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k7 ;
        kk = __shfl(k, 0 );
        ss = __shfl(seg_size, 0 );
        if(lane_id+0  <ss) valB[kk+lane_id+0  ] = val[kk+rg_v0 ];
        if(lane_id+32 <ss) valB[kk+lane_id+32 ] = val[kk+rg_v4 ];
        if(lane_id+64 <ss) valB[kk+lane_id+64 ] = val[kk+rg_v1 ];
        if(lane_id+96 <ss) valB[kk+lane_id+96 ] = val[kk+rg_v5 ];
        if(lane_id+128<ss) valB[kk+lane_id+128] = val[kk+rg_v2 ];
        if(lane_id+160<ss) valB[kk+lane_id+160] = val[kk+rg_v6 ];
        if(lane_id+192<ss) valB[kk+lane_id+192] = val[kk+rg_v3 ];
        if(lane_id+224<ss) valB[kk+lane_id+224] = val[kk+rg_v7 ];
    } else if(bin_it < bin_size) {
        if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0 ] = rg_k0 ;
        if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1 ] = rg_k1 ;
        if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2 ] = rg_k2 ;
        if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3 ] = rg_k3 ;
        if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4 ] = rg_k4 ;
        if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5 ] = rg_k5 ;
        if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6 ] = rg_k6 ;
        if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7 ] = rg_k7 ;
        if((tid<<3)+0 <seg_size) valB[k+(tid<<3)+0 ] = val[k+rg_v0 ];
        if((tid<<3)+1 <seg_size) valB[k+(tid<<3)+1 ] = val[k+rg_v1 ];
        if((tid<<3)+2 <seg_size) valB[k+(tid<<3)+2 ] = val[k+rg_v2 ];
        if((tid<<3)+3 <seg_size) valB[k+(tid<<3)+3 ] = val[k+rg_v3 ];
        if((tid<<3)+4 <seg_size) valB[k+(tid<<3)+4 ] = val[k+rg_v4 ];
        if((tid<<3)+5 <seg_size) valB[k+(tid<<3)+5 ] = val[k+rg_v5 ];
        if((tid<<3)+6 <seg_size) valB[k+(tid<<3)+6 ] = val[k+rg_v6 ];
        if((tid<<3)+7 <seg_size) valB[k+(tid<<3)+7 ] = val[k+rg_v7 ];
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   128    2    4  true      true */
__global__ void gen_bk128_tc4_r257_r512_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ int smem[512];
    __shared__ int tmem[512];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 2 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        int tmp_k0;
        int tmp_k1;
        int tmp_v0;
        int tmp_v1;
        int *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   256    2    4  true      true */
__global__ void gen_bk256_tc4_r513_r1024_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ int smem[1024];
    __shared__ int tmem[1024];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 3 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        int tmp_k0;
        int tmp_k1;
        int tmp_v0;
        int tmp_v1;
        int *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   512    2    4  true      true */
__global__ void gen_bk512_tc4_r1025_r2048_orig(int *key, int *val, int *keyB, int *valB, int n, int *segs, int *bin, int bin_size, int length)
{

    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ int smem[2048];
    __shared__ int tmem[2048];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = segs[bin[bin_it]];
        seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            rg_v0 = (warp_id<<6)+tid1+0   ;
            rg_v1 = (warp_id<<6)+tid1+32  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,
                       rg_v0 ,rg_v1 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:INT_MAX;
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:INT_MAX;
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:INT_MAX;
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:INT_MAX;
            if(sml_len+tid1+(big_warp_id<<7)+0   <seg_size) rg_v0  = sml_len+tid1+(big_warp_id<<7)+0   ;
            if(sml_len+tid1+(big_warp_id<<7)+32  <seg_size) rg_v1  = sml_len+tid1+(big_warp_id<<7)+32  ;
            if(sml_len+tid1+(big_warp_id<<7)+64  <seg_size) rg_v2  = sml_len+tid1+(big_warp_id<<7)+64  ;
            if(sml_len+tid1+(big_warp_id<<7)+96  <seg_size) rg_v3  = sml_len+tid1+(big_warp_id<<7)+96  ;
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
            CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x3,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x7,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0xf,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
            // exch_intxn: generate exch_intxn()
            exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1f,bit5);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x8,bit4);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x4,bit3);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x2,bit2);
            // exch_paral: generate exch_paral()
            exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                       rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
                       0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
            CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
            // exch_paral: switch to exch_local()
            CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
            CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
            tmem[(warp_id<<6)+(tid1<<1)+0 ] = rg_v0 ;
            tmem[(warp_id<<6)+(tid1<<1)+1 ] = rg_v1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
            tmem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Merge in 4 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        int tmp_k0;
        int tmp_k1;
        int tmp_v0;
        int tmp_v1;
        int *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
            tmem[grp_start_off+gran+0 ] = rg_v0 ;
            tmem[grp_start_off+gran+1 ] = rg_v1 ;
            tmem[grp_start_off+gran+2 ] = rg_v2 ;
            tmem[grp_start_off+gran+3 ] = rg_v3 ;
        }
        __syncthreads();
        // Step 3
        grp_start_wp_id = ((warp_id>>4)<<4);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 )+
                  ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+8<sml_wp)?64  :128 )+
                  ((tmp_wp_id+9<sml_wp)?64  :128 )+
                  ((tmp_wp_id+10<sml_wp)?64  :128 )+
                  ((tmp_wp_id+11<sml_wp)?64  :128 )+
                  ((tmp_wp_id+12<sml_wp)?64  :128 )+
                  ((tmp_wp_id+13<sml_wp)?64  :128 )+
                  ((tmp_wp_id+14<sml_wp)?64  :128 )+
                  ((tmp_wp_id+15<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&15)==0){
            gran += 0;
        }
        if((warp_id&15)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&15)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&15)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&15)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&15)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&15)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&15)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        if((warp_id&15)==8){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 );
        }
        if((warp_id&15)==9){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 );
        }
        if((warp_id&15)==10){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 );
        }
        if((warp_id&15)==11){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 );
        }
        if((warp_id&15)==12){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 );
        }
        if((warp_id&15)==13){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 );
        }
        if((warp_id&15)==14){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 );
        }
        if((warp_id&15)==15){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 )+
                    ((tmp_wp_id+14<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
            if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
            if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            rg_v0 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            rg_v1 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            rg_v2 = p ? tmp_v0 : tmp_v1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
                if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
                if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
            rg_v3 = p ? tmp_v0 : tmp_v1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
            if((tid<<1)+0 <seg_size) valB[k+(tid<<1)+0 ] = val[k+rg_v0 ];
            if((tid<<1)+1 <seg_size) valB[k+(tid<<1)+1 ] = val[k+rg_v1 ];
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
            if((tid<<2)+0 -sml_len<seg_size) valB[k+(tid<<2)+0 -sml_len] = val[k+rg_v0 ];
            if((tid<<2)+1 -sml_len<seg_size) valB[k+(tid<<2)+1 -sml_len] = val[k+rg_v1 ];
            if((tid<<2)+2 -sml_len<seg_size) valB[k+(tid<<2)+2 -sml_len] = val[k+rg_v2 ];
            if((tid<<2)+3 -sml_len<seg_size) valB[k+(tid<<2)+3 -sml_len] = val[k+rg_v3 ];
        }
    }
}
/************************* BB_COMPUT_S.H (end)  *************************/

/************************ BB_COMPUT_L.H (start)  ************************/
__device__ int binary_search(int *blk_stat, int bin_size, int gid, int blk_num)
{
    int l = 0;
    int h = bin_size;
    int m;
    int lr, rr;
    while(l < h)
    {
        m = l + (h-l)/2;
        lr = blk_stat[m];
        rr = (m==bin_size)?blk_num:blk_stat[m+1];
        if(lr<=gid && gid<rr)
        {
            return m;
        } else if(gid < lr)
        {
            h = m;
        } else
        {
            l = m+1;
        }
    }
    return m;
}

__device__ inline int upper_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

__device__ inline int log2(int u)
{
    int s, t;
    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;
    return (t | (u >> 1));
}

__global__ void kern_get_num_blk_init(int *max_segsize, int *segs, int *bin, int *blk_stat, int n, int bin_size, int length, int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        int seg_size = ((bin[gid]==length-1)?n:segs[bin[gid]+1])-segs[bin[gid]];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
        atomicMax(max_segsize, seg_size);
    }
}

__global__ void kern_get_init_pos(int *blk_stat, int *blk_innerid, int *blk_seg_start, int blk_num, int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < blk_num)
    {
        int found = binary_search(blk_stat, bin_size, gid, blk_num);
        blk_innerid[gid] = gid - blk_stat[found];
        blk_seg_start[gid] = found;
    }
}

__global__ void kern_block_sort(int *key, int *val, int *keyB, int *valB, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n)
{
    /*** codegen ***/
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];
    /*** codegen ***/
    const int tid = threadIdx.x;
    // const int bin_it = blockIdx.x;
    __shared__ int smem[2048];
    __shared__ int tmem[2048];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    int rg_k0 ;
    int rg_k1 ;
    int rg_k2 ;
    int rg_k3 ;
    int rg_v0 ;
    int rg_v1 ;
    int rg_v2 ;
    int rg_v3 ;
    // int k;
    // int ext_seg_size;
    /*** codegen ***/
    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    k = k + (innerbid<<11);
    seg_size = min(seg_size-(innerbid<<11), 2048);
    /*** codegen ***/
    rg_k0  = (tid1+(warp_id<<7)+0   <seg_size)?key[k+tid1+(warp_id<<7)+0   ]:INT_MAX;
    rg_k1  = (tid1+(warp_id<<7)+32  <seg_size)?key[k+tid1+(warp_id<<7)+32  ]:INT_MAX;
    rg_k2  = (tid1+(warp_id<<7)+64  <seg_size)?key[k+tid1+(warp_id<<7)+64  ]:INT_MAX;
    rg_k3  = (tid1+(warp_id<<7)+96  <seg_size)?key[k+tid1+(warp_id<<7)+96  ]:INT_MAX;
    if(tid1+(warp_id<<7)+0   <seg_size) rg_v0  = tid1+(warp_id<<7)+0   ;
    if(tid1+(warp_id<<7)+32  <seg_size) rg_v1  = tid1+(warp_id<<7)+32  ;
    if(tid1+(warp_id<<7)+64  <seg_size) rg_v2  = tid1+(warp_id<<7)+64  ;
    if(tid1+(warp_id<<7)+96  <seg_size) rg_v3  = tid1+(warp_id<<7)+96  ;
    // exch_intxn: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k3 ,int,rg_v0 ,rg_v3 );
    CMP_SWP(int,rg_k1 ,rg_k2 ,int,rg_v1 ,rg_v2 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x3,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x7,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0xf,bit4);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x4,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    // exch_intxn: generate exch_intxn()
    exch_intxn(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1f,bit5);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x8,bit4);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x4,bit3);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x2,bit2);
    // exch_paral: generate exch_paral()
    exch_paral(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
               rg_v0 ,rg_v1 ,rg_v2 ,rg_v3 ,
               0x1,bit1);
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k2 ,int,rg_v0 ,rg_v2 );
    CMP_SWP(int,rg_k1 ,rg_k3 ,int,rg_v1 ,rg_v3 );
    // exch_paral: switch to exch_local()
    CMP_SWP(int,rg_k0 ,rg_k1 ,int,rg_v0 ,rg_v1 );
    CMP_SWP(int,rg_k2 ,rg_k3 ,int,rg_v2 ,rg_v3 );
    
    smem[(warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
    smem[(warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
    smem[(warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
    smem[(warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
    tmem[(warp_id<<7)+(tid1<<2)+0 ] = rg_v0 ;
    tmem[(warp_id<<7)+(tid1<<2)+1 ] = rg_v1 ;
    tmem[(warp_id<<7)+(tid1<<2)+2 ] = rg_v2 ;
    tmem[(warp_id<<7)+(tid1<<2)+3 ] = rg_v3 ;
    __syncthreads();
    // Merge in 4 steps
    int grp_start_wp_id;
    int grp_start_off;
    // int tmp_wp_id;
    int lhs_len;
    int rhs_len;
    int gran;
    int s_a;
    int s_b;
    bool p;
    int tmp_k0;
    int tmp_k1;
    int tmp_v0;
    int tmp_v1;
    int *start;
    // Step 0
    grp_start_wp_id = ((warp_id>>1)<<1);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (128 );
    rhs_len = (128 );
    gran = (tid1<<2);
    if((warp_id&1)==0){
        gran += 0;
    }
    if((warp_id&1)==1){
        gran += (128 );
    }
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 1
    grp_start_wp_id = ((warp_id>>2)<<2);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (256 );
    rhs_len = (256 );
    gran = (tid1<<2);
    gran += (warp_id&3)*128;
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 2
    grp_start_wp_id = ((warp_id>>3)<<3);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (512 );
    rhs_len = (512 );
    gran = (tid1<<2);
    gran += (warp_id&7)*128;
    
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    __syncthreads();
    // Store merged results back to shared memory
    
    smem[grp_start_off+gran+0 ] = rg_k0 ;
    smem[grp_start_off+gran+1 ] = rg_k1 ;
    smem[grp_start_off+gran+2 ] = rg_k2 ;
    smem[grp_start_off+gran+3 ] = rg_k3 ;
    tmem[grp_start_off+gran+0 ] = rg_v0 ;
    tmem[grp_start_off+gran+1 ] = rg_v1 ;
    tmem[grp_start_off+gran+2 ] = rg_v2 ;
    tmem[grp_start_off+gran+3 ] = rg_v3 ;
    __syncthreads();
    // Step 3
    grp_start_wp_id = ((warp_id>>4)<<4);
    grp_start_off = (grp_start_wp_id)*128;
    // tmp_wp_id = grp_start_wp_id;
    lhs_len = (1024);
    rhs_len = (1024);
    gran = (tid1<<2);
    gran += (warp_id&15)*128;
    
    start = smem + grp_start_off;
    s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
    s_b = lhs_len + gran - s_a;
    
    tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:INT_MAX;
    tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
    if(s_a<lhs_len        ) tmp_v0 = tmem[grp_start_off+s_a];
    if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k0 = p ? tmp_k0 : tmp_k1;
    rg_v0 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k1 = p ? tmp_k0 : tmp_k1;
    rg_v1 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k2 = p ? tmp_k0 : tmp_k1;
    rg_v2 = p ? tmp_v0 : tmp_v1;
    if(p) {
        ++s_a;
        tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:INT_MAX;
        if(s_a<lhs_len) tmp_v0 = tmem[grp_start_off+s_a];
    } else {
        ++s_b;
        tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:INT_MAX;
        if(s_b<lhs_len+rhs_len) tmp_v1 = tmem[grp_start_off+s_b];
    }
    p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
    rg_k3 = p ? tmp_k0 : tmp_k1;
    rg_v3 = p ? tmp_v0 : tmp_v1;
    
    if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
    if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
    if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
    if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    int t_v0 ;
    int t_v1 ;
    int t_v2 ;
    int t_v3 ;
    if((tid<<2)+0 <seg_size) t_v0  = val[k+rg_v0 ];
    if((tid<<2)+1 <seg_size) t_v1  = val[k+rg_v1 ];
    if((tid<<2)+2 <seg_size) t_v2  = val[k+rg_v2 ];
    if((tid<<2)+3 <seg_size) t_v3  = val[k+rg_v3 ];
    if((tid<<2)+0 <seg_size) valB[k+(tid<<2)+0 ] = t_v0 ;
    if((tid<<2)+1 <seg_size) valB[k+(tid<<2)+1 ] = t_v1 ;
    if((tid<<2)+2 <seg_size) valB[k+(tid<<2)+2 ] = t_v2 ;
    if((tid<<2)+3 <seg_size) valB[k+(tid<<2)+3 ] = t_v3 ;
}

__global__ void kern_get_num_blk(int *segs, int *bin, int *blk_stat, int n, int bin_size, int length, int workloads_per_block)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid < bin_size)
    {
        int seg_size = ((bin[gid]==length-1)?n:segs[bin[gid]+1])-segs[bin[gid]];
        blk_stat[gid] = (seg_size+workloads_per_block-1)/workloads_per_block;
    }
}

__global__ void kern_block_merge(int *keys, int *vals, int *keysB, int *valsB, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n, int stride)
{
    __shared__ int smem[128*16];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];

    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    if(stride < seg_size)
    {
        int loc_a, loc_b;
        int cnt_a, cnt_b;
        int coop = (stride<<1)>>11; // how many blocks coop
        int coop_bid = innerbid%coop;
        int l_gran, r_gran;
        loc_a = (innerbid/coop)*(stride<<1);
        cnt_a = min(stride, seg_size-loc_a);
        loc_b = min(loc_a + stride, seg_size);
        cnt_b = min(stride, seg_size-loc_b);
        l_gran = coop_bid<<11;
        r_gran = min((coop_bid+1)<<11, seg_size-loc_a);
        int l_s_a, l_s_b;
        int r_s_a, r_s_b;
        l_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, l_gran);
        l_s_b = l_gran - l_s_a;
        r_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, r_gran);
        r_s_b = r_gran - r_s_a;
        int l_st = 0;
        int l_cnt = r_s_a - l_s_a;
        if(l_s_a+tid     <r_s_a) smem[l_st+tid     ] = keys[k+loc_a+l_s_a+tid     ];
        if(l_s_a+tid+128 <r_s_a) smem[l_st+tid+128 ] = keys[k+loc_a+l_s_a+tid+128 ];
        if(l_s_a+tid+256 <r_s_a) smem[l_st+tid+256 ] = keys[k+loc_a+l_s_a+tid+256 ];
        if(l_s_a+tid+384 <r_s_a) smem[l_st+tid+384 ] = keys[k+loc_a+l_s_a+tid+384 ];
        if(l_s_a+tid+512 <r_s_a) smem[l_st+tid+512 ] = keys[k+loc_a+l_s_a+tid+512 ];
        if(l_s_a+tid+640 <r_s_a) smem[l_st+tid+640 ] = keys[k+loc_a+l_s_a+tid+640 ];
        if(l_s_a+tid+768 <r_s_a) smem[l_st+tid+768 ] = keys[k+loc_a+l_s_a+tid+768 ];
        if(l_s_a+tid+896 <r_s_a) smem[l_st+tid+896 ] = keys[k+loc_a+l_s_a+tid+896 ];
        if(l_s_a+tid+1024<r_s_a) smem[l_st+tid+1024] = keys[k+loc_a+l_s_a+tid+1024];
        if(l_s_a+tid+1152<r_s_a) smem[l_st+tid+1152] = keys[k+loc_a+l_s_a+tid+1152];
        if(l_s_a+tid+1280<r_s_a) smem[l_st+tid+1280] = keys[k+loc_a+l_s_a+tid+1280];
        if(l_s_a+tid+1408<r_s_a) smem[l_st+tid+1408] = keys[k+loc_a+l_s_a+tid+1408];
        if(l_s_a+tid+1536<r_s_a) smem[l_st+tid+1536] = keys[k+loc_a+l_s_a+tid+1536];
        if(l_s_a+tid+1664<r_s_a) smem[l_st+tid+1664] = keys[k+loc_a+l_s_a+tid+1664];
        if(l_s_a+tid+1792<r_s_a) smem[l_st+tid+1792] = keys[k+loc_a+l_s_a+tid+1792];
        if(l_s_a+tid+1920<r_s_a) smem[l_st+tid+1920] = keys[k+loc_a+l_s_a+tid+1920];
        int r_st = r_s_a - l_s_a;
        int r_cnt = r_s_b - l_s_b;
        if(l_s_b+tid     <r_s_b) smem[r_st+tid     ] = keys[k+loc_b+l_s_b+tid     ];
        if(l_s_b+tid+128 <r_s_b) smem[r_st+tid+128 ] = keys[k+loc_b+l_s_b+tid+128 ];
        if(l_s_b+tid+256 <r_s_b) smem[r_st+tid+256 ] = keys[k+loc_b+l_s_b+tid+256 ];
        if(l_s_b+tid+384 <r_s_b) smem[r_st+tid+384 ] = keys[k+loc_b+l_s_b+tid+384 ];
        if(l_s_b+tid+512 <r_s_b) smem[r_st+tid+512 ] = keys[k+loc_b+l_s_b+tid+512 ];
        if(l_s_b+tid+640 <r_s_b) smem[r_st+tid+640 ] = keys[k+loc_b+l_s_b+tid+640 ];
        if(l_s_b+tid+768 <r_s_b) smem[r_st+tid+768 ] = keys[k+loc_b+l_s_b+tid+768 ];
        if(l_s_b+tid+896 <r_s_b) smem[r_st+tid+896 ] = keys[k+loc_b+l_s_b+tid+896 ];
        if(l_s_b+tid+1024<r_s_b) smem[r_st+tid+1024] = keys[k+loc_b+l_s_b+tid+1024];
        if(l_s_b+tid+1152<r_s_b) smem[r_st+tid+1152] = keys[k+loc_b+l_s_b+tid+1152];
        if(l_s_b+tid+1280<r_s_b) smem[r_st+tid+1280] = keys[k+loc_b+l_s_b+tid+1280];
        if(l_s_b+tid+1408<r_s_b) smem[r_st+tid+1408] = keys[k+loc_b+l_s_b+tid+1408];
        if(l_s_b+tid+1536<r_s_b) smem[r_st+tid+1536] = keys[k+loc_b+l_s_b+tid+1536];
        if(l_s_b+tid+1664<r_s_b) smem[r_st+tid+1664] = keys[k+loc_b+l_s_b+tid+1664];
        if(l_s_b+tid+1792<r_s_b) smem[r_st+tid+1792] = keys[k+loc_b+l_s_b+tid+1792];
        if(l_s_b+tid+1920<r_s_b) smem[r_st+tid+1920] = keys[k+loc_b+l_s_b+tid+1920];
        __syncthreads();

        int gran = tid<<4;
        int s_a, s_b;
        bool p;
        int rg_k0 ;
        int rg_k1 ;
        int rg_k2 ;
        int rg_k3 ;
        int rg_k4 ;
        int rg_k5 ;
        int rg_k6 ;
        int rg_k7 ;
        int rg_k8 ;
        int rg_k9 ;
        int rg_k10;
        int rg_k11;
        int rg_k12;
        int rg_k13;
        int rg_k14;
        int rg_k15;
        int rg_v0 ;
        int rg_v1 ;
        int rg_v2 ;
        int rg_v3 ;
        int rg_v4 ;
        int rg_v5 ;
        int rg_v6 ;
        int rg_v7 ;
        int rg_v8 ;
        int rg_v9 ;
        int rg_v10;
        int rg_v11;
        int rg_v12;
        int rg_v13;
        int rg_v14;
        int rg_v15;
        int tmp_k0,tmp_k1;
        int tmp_v0,tmp_v1;

        s_a = find_kth3(smem+l_st, l_cnt, smem+r_st, r_cnt, gran);
        s_b = gran - s_a;
        tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
        tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
        if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k0 = p ? tmp_k0 : tmp_k1;
        rg_v0 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k1 = p ? tmp_k0 : tmp_k1;
        rg_v1 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k2 = p ? tmp_k0 : tmp_k1;
        rg_v2 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k3 = p ? tmp_k0 : tmp_k1;
        rg_v3 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k4 = p ? tmp_k0 : tmp_k1;
        rg_v4 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k5 = p ? tmp_k0 : tmp_k1;
        rg_v5 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k6 = p ? tmp_k0 : tmp_k1;
        rg_v6 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k7 = p ? tmp_k0 : tmp_k1;
        rg_v7 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k8 = p ? tmp_k0 : tmp_k1;
        rg_v8 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k9 = p ? tmp_k0 : tmp_k1;
        rg_v9 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k10 = p ? tmp_k0 : tmp_k1;
        rg_v10 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k11 = p ? tmp_k0 : tmp_k1;
        rg_v11 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k12 = p ? tmp_k0 : tmp_k1;
        rg_v12 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k13 = p ? tmp_k0 : tmp_k1;
        rg_v13 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k14 = p ? tmp_k0 : tmp_k1;
        rg_v14 = p ? tmp_v0 : tmp_v1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:INT_MAX;
            if(s_a < l_cnt) tmp_v0 = (loc_a+l_s_a+s_a);
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:INT_MAX;
            if(s_b < r_cnt) tmp_v1 = (loc_b+l_s_b+s_b);
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k15 = p ? tmp_k0 : tmp_k1;
        rg_v15 = p ? tmp_v0 : tmp_v1;

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        if(lane_id&0x1) SWP(int, rg_k0 ,rg_k1 , int, rg_v0 ,rg_v1 );
        if(lane_id&0x1) SWP(int, rg_k2 ,rg_k3 , int, rg_v2 ,rg_v3 );
        if(lane_id&0x1) SWP(int, rg_k4 ,rg_k5 , int, rg_v4 ,rg_v5 );
        if(lane_id&0x1) SWP(int, rg_k6 ,rg_k7 , int, rg_v6 ,rg_v7 );
        if(lane_id&0x1) SWP(int, rg_k8 ,rg_k9 , int, rg_v8 ,rg_v9 );
        if(lane_id&0x1) SWP(int, rg_k10,rg_k11, int, rg_v10,rg_v11);
        if(lane_id&0x1) SWP(int, rg_k12,rg_k13, int, rg_v12,rg_v13);
        if(lane_id&0x1) SWP(int, rg_k14,rg_k15, int, rg_v14,rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x1 );
        rg_k3  = __shfl_xor(rg_k3 , 0x1 );
        rg_k5  = __shfl_xor(rg_k5 , 0x1 );
        rg_k7  = __shfl_xor(rg_k7 , 0x1 );
        rg_k9  = __shfl_xor(rg_k9 , 0x1 );
        rg_k11 = __shfl_xor(rg_k11, 0x1 );
        rg_k13 = __shfl_xor(rg_k13, 0x1 );
        rg_k15 = __shfl_xor(rg_k15, 0x1 );
        rg_v1  = __shfl_xor(rg_v1 , 0x1 );
        rg_v3  = __shfl_xor(rg_v3 , 0x1 );
        rg_v5  = __shfl_xor(rg_v5 , 0x1 );
        rg_v7  = __shfl_xor(rg_v7 , 0x1 );
        rg_v9  = __shfl_xor(rg_v9 , 0x1 );
        rg_v11 = __shfl_xor(rg_v11, 0x1 );
        rg_v13 = __shfl_xor(rg_v13, 0x1 );
        rg_v15 = __shfl_xor(rg_v15, 0x1 );
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        if(lane_id&0x2 ) SWP(int, rg_k0 , rg_k2 , int, rg_v0 , rg_v2 );
        if(lane_id&0x2 ) SWP(int, rg_k1 , rg_k3 , int, rg_v1 , rg_v3 );
        if(lane_id&0x2 ) SWP(int, rg_k4 , rg_k6 , int, rg_v4 , rg_v6 );
        if(lane_id&0x2 ) SWP(int, rg_k5 , rg_k7 , int, rg_v5 , rg_v7 );
        if(lane_id&0x2 ) SWP(int, rg_k8 , rg_k10, int, rg_v8 , rg_v10);
        if(lane_id&0x2 ) SWP(int, rg_k9 , rg_k11, int, rg_v9 , rg_v11);
        if(lane_id&0x2 ) SWP(int, rg_k12, rg_k14, int, rg_v12, rg_v14);
        if(lane_id&0x2 ) SWP(int, rg_k13, rg_k15, int, rg_v13, rg_v15);
        rg_k2  = __shfl_xor(rg_k2 , 0x2 );
        rg_k3  = __shfl_xor(rg_k3 , 0x2 );
        rg_k6  = __shfl_xor(rg_k6 , 0x2 );
        rg_k7  = __shfl_xor(rg_k7 , 0x2 );
        rg_k10 = __shfl_xor(rg_k10, 0x2 );
        rg_k11 = __shfl_xor(rg_k11, 0x2 );
        rg_k14 = __shfl_xor(rg_k14, 0x2 );
        rg_k15 = __shfl_xor(rg_k15, 0x2 );
        rg_v2  = __shfl_xor(rg_v2 , 0x2 );
        rg_v3  = __shfl_xor(rg_v3 , 0x2 );
        rg_v6  = __shfl_xor(rg_v6 , 0x2 );
        rg_v7  = __shfl_xor(rg_v7 , 0x2 );
        rg_v10 = __shfl_xor(rg_v10, 0x2 );
        rg_v11 = __shfl_xor(rg_v11, 0x2 );
        rg_v14 = __shfl_xor(rg_v14, 0x2 );
        rg_v15 = __shfl_xor(rg_v15, 0x2 );
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        if(lane_id&0x4 ) SWP(int, rg_k0 , rg_k4 , int, rg_v0 , rg_v4 );
        if(lane_id&0x4 ) SWP(int, rg_k1 , rg_k5 , int, rg_v1 , rg_v5 );
        if(lane_id&0x4 ) SWP(int, rg_k2 , rg_k6 , int, rg_v2 , rg_v6 );
        if(lane_id&0x4 ) SWP(int, rg_k3 , rg_k7 , int, rg_v3 , rg_v7 );
        if(lane_id&0x4 ) SWP(int, rg_k8 , rg_k12, int, rg_v8 , rg_v12);
        if(lane_id&0x4 ) SWP(int, rg_k9 , rg_k13, int, rg_v9 , rg_v13);
        if(lane_id&0x4 ) SWP(int, rg_k10, rg_k14, int, rg_v10, rg_v14);
        if(lane_id&0x4 ) SWP(int, rg_k11, rg_k15, int, rg_v11, rg_v15);
        rg_k4  = __shfl_xor(rg_k4 , 0x4 );
        rg_k5  = __shfl_xor(rg_k5 , 0x4 );
        rg_k6  = __shfl_xor(rg_k6 , 0x4 );
        rg_k7  = __shfl_xor(rg_k7 , 0x4 );
        rg_k12 = __shfl_xor(rg_k12, 0x4 );
        rg_k13 = __shfl_xor(rg_k13, 0x4 );
        rg_k14 = __shfl_xor(rg_k14, 0x4 );
        rg_k15 = __shfl_xor(rg_k15, 0x4 );
        rg_v4  = __shfl_xor(rg_v4 , 0x4 );
        rg_v5  = __shfl_xor(rg_v5 , 0x4 );
        rg_v6  = __shfl_xor(rg_v6 , 0x4 );
        rg_v7  = __shfl_xor(rg_v7 , 0x4 );
        rg_v12 = __shfl_xor(rg_v12, 0x4 );
        rg_v13 = __shfl_xor(rg_v13, 0x4 );
        rg_v14 = __shfl_xor(rg_v14, 0x4 );
        rg_v15 = __shfl_xor(rg_v15, 0x4 );
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        if(lane_id&0x8 ) SWP(int, rg_k0 , rg_k8 , int, rg_v0 , rg_v8 );
        if(lane_id&0x8 ) SWP(int, rg_k1 , rg_k9 , int, rg_v1 , rg_v9 );
        if(lane_id&0x8 ) SWP(int, rg_k2 , rg_k10, int, rg_v2 , rg_v10);
        if(lane_id&0x8 ) SWP(int, rg_k3 , rg_k11, int, rg_v3 , rg_v11);
        if(lane_id&0x8 ) SWP(int, rg_k4 , rg_k12, int, rg_v4 , rg_v12);
        if(lane_id&0x8 ) SWP(int, rg_k5 , rg_k13, int, rg_v5 , rg_v13);
        if(lane_id&0x8 ) SWP(int, rg_k6 , rg_k14, int, rg_v6 , rg_v14);
        if(lane_id&0x8 ) SWP(int, rg_k7 , rg_k15, int, rg_v7 , rg_v15);
        rg_k8  = __shfl_xor(rg_k8 , 0x8 );
        rg_k9  = __shfl_xor(rg_k9 , 0x8 );
        rg_k10 = __shfl_xor(rg_k10, 0x8 );
        rg_k11 = __shfl_xor(rg_k11, 0x8 );
        rg_k12 = __shfl_xor(rg_k12, 0x8 );
        rg_k13 = __shfl_xor(rg_k13, 0x8 );
        rg_k14 = __shfl_xor(rg_k14, 0x8 );
        rg_k15 = __shfl_xor(rg_k15, 0x8 );
        rg_v8  = __shfl_xor(rg_v8 , 0x8 );
        rg_v9  = __shfl_xor(rg_v9 , 0x8 );
        rg_v10 = __shfl_xor(rg_v10, 0x8 );
        rg_v11 = __shfl_xor(rg_v11, 0x8 );
        rg_v12 = __shfl_xor(rg_v12, 0x8 );
        rg_v13 = __shfl_xor(rg_v13, 0x8 );
        rg_v14 = __shfl_xor(rg_v14, 0x8 );
        rg_v15 = __shfl_xor(rg_v15, 0x8 );
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);
        if(lane_id&0x10) SWP(int, rg_k0 , rg_k1 , int, rg_v0 , rg_v1 );
        if(lane_id&0x10) SWP(int, rg_k2 , rg_k3 , int, rg_v2 , rg_v3 );
        if(lane_id&0x10) SWP(int, rg_k4 , rg_k5 , int, rg_v4 , rg_v5 );
        if(lane_id&0x10) SWP(int, rg_k6 , rg_k7 , int, rg_v6 , rg_v7 );
        if(lane_id&0x10) SWP(int, rg_k8 , rg_k9 , int, rg_v8 , rg_v9 );
        if(lane_id&0x10) SWP(int, rg_k10, rg_k11, int, rg_v10, rg_v11);
        if(lane_id&0x10) SWP(int, rg_k12, rg_k13, int, rg_v12, rg_v13);
        if(lane_id&0x10) SWP(int, rg_k14, rg_k15, int, rg_v14, rg_v15);
        rg_k1  = __shfl_xor(rg_k1 , 0x10);
        rg_k3  = __shfl_xor(rg_k3 , 0x10);
        rg_k5  = __shfl_xor(rg_k5 , 0x10);
        rg_k7  = __shfl_xor(rg_k7 , 0x10);
        rg_k9  = __shfl_xor(rg_k9 , 0x10);
        rg_k11 = __shfl_xor(rg_k11, 0x10);
        rg_k13 = __shfl_xor(rg_k13, 0x10);
        rg_k15 = __shfl_xor(rg_k15, 0x10);
        rg_v1  = __shfl_xor(rg_v1 , 0x10);
        rg_v3  = __shfl_xor(rg_v3 , 0x10);
        rg_v5  = __shfl_xor(rg_v5 , 0x10);
        rg_v7  = __shfl_xor(rg_v7 , 0x10);
        rg_v9  = __shfl_xor(rg_v9 , 0x10);
        rg_v11 = __shfl_xor(rg_v11, 0x10);
        rg_v13 = __shfl_xor(rg_v13, 0x10);
        rg_v15 = __shfl_xor(rg_v15, 0x10);

        if((innerbid<<11)+(warp_id<<9)+0  +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+0  +lane_id] = rg_k0 ;
        if((innerbid<<11)+(warp_id<<9)+32 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+32 +lane_id] = rg_k2 ;
        if((innerbid<<11)+(warp_id<<9)+64 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+64 +lane_id] = rg_k4 ;
        if((innerbid<<11)+(warp_id<<9)+96 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+96 +lane_id] = rg_k6 ;
        if((innerbid<<11)+(warp_id<<9)+128+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+128+lane_id] = rg_k8 ;
        if((innerbid<<11)+(warp_id<<9)+160+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+160+lane_id] = rg_k10;
        if((innerbid<<11)+(warp_id<<9)+192+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+192+lane_id] = rg_k12;
        if((innerbid<<11)+(warp_id<<9)+224+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+224+lane_id] = rg_k14;
        if((innerbid<<11)+(warp_id<<9)+256+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+256+lane_id] = rg_k1 ;
        if((innerbid<<11)+(warp_id<<9)+288+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+288+lane_id] = rg_k3 ;
        if((innerbid<<11)+(warp_id<<9)+320+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+320+lane_id] = rg_k5 ;
        if((innerbid<<11)+(warp_id<<9)+352+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+352+lane_id] = rg_k7 ;
        if((innerbid<<11)+(warp_id<<9)+384+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+384+lane_id] = rg_k9 ;
        if((innerbid<<11)+(warp_id<<9)+416+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+416+lane_id] = rg_k11;
        if((innerbid<<11)+(warp_id<<9)+448+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+448+lane_id] = rg_k13;
        if((innerbid<<11)+(warp_id<<9)+480+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+480+lane_id] = rg_k15;

        if((innerbid<<11)+(warp_id<<9)+0  +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+0  +lane_id]=vals[k+rg_v0 ];
        if((innerbid<<11)+(warp_id<<9)+32 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+32 +lane_id]=vals[k+rg_v2 ];
        if((innerbid<<11)+(warp_id<<9)+64 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+64 +lane_id]=vals[k+rg_v4 ];
        if((innerbid<<11)+(warp_id<<9)+96 +lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+96 +lane_id]=vals[k+rg_v6 ];
        if((innerbid<<11)+(warp_id<<9)+128+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+128+lane_id]=vals[k+rg_v8 ];
        if((innerbid<<11)+(warp_id<<9)+160+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+160+lane_id]=vals[k+rg_v10];
        if((innerbid<<11)+(warp_id<<9)+192+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+192+lane_id]=vals[k+rg_v12];
        if((innerbid<<11)+(warp_id<<9)+224+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+224+lane_id]=vals[k+rg_v14];
        if((innerbid<<11)+(warp_id<<9)+256+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+256+lane_id]=vals[k+rg_v1 ];
        if((innerbid<<11)+(warp_id<<9)+288+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+288+lane_id]=vals[k+rg_v3 ];
        if((innerbid<<11)+(warp_id<<9)+320+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+320+lane_id]=vals[k+rg_v5 ];
        if((innerbid<<11)+(warp_id<<9)+352+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+352+lane_id]=vals[k+rg_v7 ];
        if((innerbid<<11)+(warp_id<<9)+384+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+384+lane_id]=vals[k+rg_v9 ];
        if((innerbid<<11)+(warp_id<<9)+416+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+416+lane_id]=vals[k+rg_v11];
        if((innerbid<<11)+(warp_id<<9)+448+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+448+lane_id]=vals[k+rg_v13];
        if((innerbid<<11)+(warp_id<<9)+480+lane_id<seg_size)valsB[k+(innerbid<<11)+(warp_id<<9)+480+lane_id]=vals[k+rg_v15];
    }
}

__global__ void kern_copy(int *srck, int *srcv, int *dstk, int *dstv, int *segs, int *bin, int *blk_innerid, int *blk_seg_start, int length, int n, int res)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int innerbid = blk_innerid[bid];
    int bin_it  = blk_seg_start[bid];
    int k = segs[bin[bin_it]];
    int seg_size = ((bin[bin_it]==length-1)?n:segs[bin[bin_it]+1])-segs[bin[bin_it]];
    int stride = upper_power_of_two(seg_size);
    int steps = log2(stride/2048);

    if((steps&1))
    {
        if((innerbid<<11)+tid     <seg_size) dstk[k+(innerbid<<11)+tid     ] = srck[k+(innerbid<<11)+tid     ];
        if((innerbid<<11)+tid+128 <seg_size) dstk[k+(innerbid<<11)+tid+128 ] = srck[k+(innerbid<<11)+tid+128 ];
        if((innerbid<<11)+tid+256 <seg_size) dstk[k+(innerbid<<11)+tid+256 ] = srck[k+(innerbid<<11)+tid+256 ];
        if((innerbid<<11)+tid+384 <seg_size) dstk[k+(innerbid<<11)+tid+384 ] = srck[k+(innerbid<<11)+tid+384 ];
        if((innerbid<<11)+tid+512 <seg_size) dstk[k+(innerbid<<11)+tid+512 ] = srck[k+(innerbid<<11)+tid+512 ];
        if((innerbid<<11)+tid+640 <seg_size) dstk[k+(innerbid<<11)+tid+640 ] = srck[k+(innerbid<<11)+tid+640 ];
        if((innerbid<<11)+tid+768 <seg_size) dstk[k+(innerbid<<11)+tid+768 ] = srck[k+(innerbid<<11)+tid+768 ];
        if((innerbid<<11)+tid+896 <seg_size) dstk[k+(innerbid<<11)+tid+896 ] = srck[k+(innerbid<<11)+tid+896 ];
        if((innerbid<<11)+tid+1024<seg_size) dstk[k+(innerbid<<11)+tid+1024] = srck[k+(innerbid<<11)+tid+1024];
        if((innerbid<<11)+tid+1152<seg_size) dstk[k+(innerbid<<11)+tid+1152] = srck[k+(innerbid<<11)+tid+1152];
        if((innerbid<<11)+tid+1280<seg_size) dstk[k+(innerbid<<11)+tid+1280] = srck[k+(innerbid<<11)+tid+1280];
        if((innerbid<<11)+tid+1408<seg_size) dstk[k+(innerbid<<11)+tid+1408] = srck[k+(innerbid<<11)+tid+1408];
        if((innerbid<<11)+tid+1536<seg_size) dstk[k+(innerbid<<11)+tid+1536] = srck[k+(innerbid<<11)+tid+1536];
        if((innerbid<<11)+tid+1664<seg_size) dstk[k+(innerbid<<11)+tid+1664] = srck[k+(innerbid<<11)+tid+1664];
        if((innerbid<<11)+tid+1792<seg_size) dstk[k+(innerbid<<11)+tid+1792] = srck[k+(innerbid<<11)+tid+1792];
        if((innerbid<<11)+tid+1920<seg_size) dstk[k+(innerbid<<11)+tid+1920] = srck[k+(innerbid<<11)+tid+1920];

        if((innerbid<<11)+tid     <seg_size) dstv[k+(innerbid<<11)+tid     ] = srcv[k+(innerbid<<11)+tid     ];
        if((innerbid<<11)+tid+128 <seg_size) dstv[k+(innerbid<<11)+tid+128 ] = srcv[k+(innerbid<<11)+tid+128 ];
        if((innerbid<<11)+tid+256 <seg_size) dstv[k+(innerbid<<11)+tid+256 ] = srcv[k+(innerbid<<11)+tid+256 ];
        if((innerbid<<11)+tid+384 <seg_size) dstv[k+(innerbid<<11)+tid+384 ] = srcv[k+(innerbid<<11)+tid+384 ];
        if((innerbid<<11)+tid+512 <seg_size) dstv[k+(innerbid<<11)+tid+512 ] = srcv[k+(innerbid<<11)+tid+512 ];
        if((innerbid<<11)+tid+640 <seg_size) dstv[k+(innerbid<<11)+tid+640 ] = srcv[k+(innerbid<<11)+tid+640 ];
        if((innerbid<<11)+tid+768 <seg_size) dstv[k+(innerbid<<11)+tid+768 ] = srcv[k+(innerbid<<11)+tid+768 ];
        if((innerbid<<11)+tid+896 <seg_size) dstv[k+(innerbid<<11)+tid+896 ] = srcv[k+(innerbid<<11)+tid+896 ];
        if((innerbid<<11)+tid+1024<seg_size) dstv[k+(innerbid<<11)+tid+1024] = srcv[k+(innerbid<<11)+tid+1024];
        if((innerbid<<11)+tid+1152<seg_size) dstv[k+(innerbid<<11)+tid+1152] = srcv[k+(innerbid<<11)+tid+1152];
        if((innerbid<<11)+tid+1280<seg_size) dstv[k+(innerbid<<11)+tid+1280] = srcv[k+(innerbid<<11)+tid+1280];
        if((innerbid<<11)+tid+1408<seg_size) dstv[k+(innerbid<<11)+tid+1408] = srcv[k+(innerbid<<11)+tid+1408];
        if((innerbid<<11)+tid+1536<seg_size) dstv[k+(innerbid<<11)+tid+1536] = srcv[k+(innerbid<<11)+tid+1536];
        if((innerbid<<11)+tid+1664<seg_size) dstv[k+(innerbid<<11)+tid+1664] = srcv[k+(innerbid<<11)+tid+1664];
        if((innerbid<<11)+tid+1792<seg_size) dstv[k+(innerbid<<11)+tid+1792] = srcv[k+(innerbid<<11)+tid+1792];
        if((innerbid<<11)+tid+1920<seg_size) dstv[k+(innerbid<<11)+tid+1920] = srcv[k+(innerbid<<11)+tid+1920];
    }
}

int gen_grid_kern_r2049(int *keys_d, int *vals_d, int *keysB_d, int *valsB_d, int n, int *segs_d, int *bin_d, int bin_size, int length)
{
    cudaError_t err;
    int *blk_stat_d; // histogram of how many blocks for each seg
    err = cudaMalloc((void **)&blk_stat_d, bin_size*sizeof(int));
    ERR_INFO(err, "alloc blk_stat_d");

    int blk_num; // total number of blocks
    int *max_segsize_d;
    err = cudaMalloc((void **)&max_segsize_d, sizeof(int));
    ERR_INFO(err, "alloc max_segsize_d");

    err = cudaMemset(max_segsize_d, 0, sizeof(int));
    ERR_INFO(err, "memset max_segsize_d");

    dim3 blocks(256, 1, 1);
    dim3 grids((bin_size+blocks.x-1)/blocks.x, 1, 1);

    // this kernel gets how many blocks for each seg; get max seg length;
    // last parameter is how many pairs one block can handle
    kern_get_num_blk_init<<<grids, blocks>>>(max_segsize_d, segs_d, bin_d, blk_stat_d, 
            n, bin_size, length, 2048); // 512thread*4key /*** codegen ***/

    int max_segsize;
    err = cudaMemcpy(&max_segsize, max_segsize_d, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from max_segsize_d");
    // store the last number from blk_stat_d in case of the exclusive scan later on
    err = cudaMemcpy(&blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  

    thrust::device_ptr<int> d_arr0 = thrust::device_pointer_cast<int>(blk_stat_d);
    thrust::exclusive_scan(d_arr0, d_arr0+bin_size, d_arr0);

    int part_blk_num;
    err = cudaMemcpy(&part_blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  
    blk_num = blk_num + part_blk_num;

    int *blk_innerid; // record each block's inner id
    err = cudaMalloc((void **)&blk_innerid, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_innerid");  

    int *blk_seg_start; // record each block's segment's starting position
    err = cudaMalloc((void **)&blk_seg_start, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_seg_start");  

    grids.x = (blk_num+blocks.x-1)/blocks.x;
    kern_get_init_pos<<<grids, blocks>>>(blk_stat_d, blk_innerid, blk_seg_start, 
            blk_num, bin_size);

    /*** codegen ***/
    blocks.x = 512;
    grids.x = blk_num;
    kern_block_sort<<<grids, blocks>>>(keys_d, vals_d, keysB_d, valsB_d, segs_d, bin_d, 
            blk_innerid, blk_seg_start, length, n);

    blocks.x = 256;
    grids.x = (bin_size+blocks.x-1)/blocks.x;
    kern_get_num_blk<<<grids, blocks>>>(segs_d, bin_d, blk_stat_d, 
            n, bin_size, length, 2048); // 128t*16k /*** codegen ***/

    err = cudaMemcpy(&blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  

    thrust::device_ptr<int> d_arr1 = thrust::device_pointer_cast<int>(blk_stat_d);
    thrust::exclusive_scan(d_arr1, d_arr1+bin_size, d_arr1);

    err = cudaMemcpy(&part_blk_num, blk_stat_d+bin_size-1, sizeof(int), cudaMemcpyDeviceToHost);
    ERR_INFO(err, "copy from blk_stat_d+bin_size-1");  
    blk_num = blk_num + part_blk_num;

    err = cudaFree(blk_innerid);
    ERR_INFO(err, "free blk_innerid");  

    err = cudaMalloc((void **)&blk_innerid, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_innerid");  

    err = cudaFree(blk_seg_start);
    ERR_INFO(err, "free blk_seg_start");  

    err = cudaMalloc((void **)&blk_seg_start, blk_num*sizeof(int));
    ERR_INFO(err, "alloc blk_seg_start");  

    grids.x = (blk_num+blocks.x-1)/blocks.x;
    kern_get_init_pos<<<grids, blocks>>>(blk_stat_d, blk_innerid, blk_seg_start, 
            blk_num, bin_size);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    int stride = 2048; // unit for already sorted
    int cnt = 0;
    blocks.x = 128;
    grids.x = blk_num;

    // cout << "max_segsize " << max_segsize << endl;
    while(stride < max_segsize)
    {
        kern_block_merge<<<grids, blocks>>>(keys_d, vals_d, keysB_d, valsB_d, segs_d, bin_d, 
                blk_innerid, blk_seg_start, length, n, stride);
        stride <<= 1;
        std::swap(keys_d, keysB_d);
        std::swap(vals_d, valsB_d);
        cnt++;
    }

    // cout << "cnt " << cnt << endl;

    blocks.x = 128;
    grids.x = blk_num;
    int *srck = (cnt&1)?keys_d:keysB_d;
    int *dstk = (cnt&1)?keysB_d:keys_d;
    int *srcv = (cnt&1)?vals_d:valsB_d;
    int *dstv = (cnt&1)?valsB_d:vals_d;
    kern_copy<<<grids, blocks>>>(srck, srcv, dstk, dstv, segs_d, bin_d, 
                blk_innerid, blk_seg_start, length, n, cnt);
    
    err = cudaFree(blk_stat_d);
    ERR_INFO(err, "free blk_stat_d");  
    err = cudaFree(blk_innerid);
    ERR_INFO(err, "free blk_innerid");  
    err = cudaFree(blk_seg_start);
    ERR_INFO(err, "free blk_seg_start");  
    err = cudaFree(max_segsize_d);
    ERR_INFO(err, "free max_segsize_d");  

    return cnt-1;
}
/************************* BB_COMPUT_L.H (end)  *************************/

/************************** BB_BIN.H (start)   **************************/
void bb_bin(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, const int length, const int n, int *h_bin_counter)
{
    const int num_threads = 256;
    const int num_blocks = ceil((double)length/(double)num_threads);

    bb_bin_histo<<< num_blocks, num_threads >>>(d_bin_counter, d_segs, length, n);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    thrust::device_ptr<int> d_arr = thrust::device_pointer_cast<int>(d_bin_counter);
    thrust::exclusive_scan(d_arr, d_arr + SEGBIN_NUM, d_arr);

    // show_d(d_bin_counter, SEGBIN_NUM, "d_bin_counter:\n");

    cudaMemcpyAsync(h_bin_counter, d_bin_counter, SEGBIN_NUM*sizeof(int), cudaMemcpyDeviceToHost);

    // group segment IDs (that belong to the same bin) together
    bb_bin_group<<< num_blocks, num_threads >>>(d_bin_segs_id, d_bin_counter, d_segs, length, n);

    // show_d(d_bin_segs_id, length, "d_bin_segs_id:\n");
}

__global__ void bb_bin_histo(int *d_bin_counter, const int *d_segs, int length, int n)
{
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x; 

    __shared__ int local_histo[SEGBIN_NUM];
    if (tid < SEGBIN_NUM)
        local_histo[tid] = 0;
    __syncthreads();

    if (gid < length)
    {
        const int size = ((gid==length-1)?n:d_segs[gid+1]) - d_segs[gid];

        if (size <= 1)
            atomicAdd((int *)&local_histo[0 ], 1);
        if (1  < size && size <= 2 )
            atomicAdd((int *)&local_histo[1 ], 1);
        if (2  < size && size <= 4 )
            atomicAdd((int *)&local_histo[2 ], 1);
        if (4  < size && size <= 8 )
            atomicAdd((int *)&local_histo[3 ], 1);
        if (8  < size && size <= 16)
            atomicAdd((int *)&local_histo[4 ], 1);
        if (16 < size && size <= 32)
            atomicAdd((int *)&local_histo[5 ], 1);
        if (32 < size && size <= 64)
            atomicAdd((int *)&local_histo[6 ], 1);
        if (64 < size && size <= 128)
            atomicAdd((int *)&local_histo[7 ], 1);
        if (128 < size && size <= 256)
            atomicAdd((int *)&local_histo[8 ], 1);
        if (256 < size && size <= 512)
            atomicAdd((int *)&local_histo[9 ], 1);
        if (512 < size && size <= 1024)
            atomicAdd((int *)&local_histo[10], 1);
        if (1024 < size && size <= 2048)
            atomicAdd((int *)&local_histo[11], 1);
        if (2048 < size)
            atomicAdd((int *)&local_histo[12], 1);
    }
    __syncthreads();

    if (tid < SEGBIN_NUM)
        atomicAdd((int *)&d_bin_counter[tid], local_histo[tid]);
}

__global__ void bb_bin_group(int *d_bin_segs_id, int *d_bin_counter, const int *d_segs, int length, int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < length)
    {
        const int size = ((gid==length-1)?n:d_segs[gid+1]) - d_segs[gid];
        int position;
        if (size <= 1)
            position = atomicAdd((int *)&d_bin_counter[0 ], 1);
        else if (size <= 2)                              
            position = atomicAdd((int *)&d_bin_counter[1 ], 1);
        else if (size <= 4)                              
            position = atomicAdd((int *)&d_bin_counter[2 ], 1);
        else if (size <= 8)                              
            position = atomicAdd((int *)&d_bin_counter[3 ], 1);
        else if (size <= 16)                             
            position = atomicAdd((int *)&d_bin_counter[4 ], 1);
        else if (size <= 32)                             
            position = atomicAdd((int *)&d_bin_counter[5 ], 1);
        else if (size <= 64)                             
            position = atomicAdd((int *)&d_bin_counter[6 ], 1);
        else if (size <= 128)                            
            position = atomicAdd((int *)&d_bin_counter[7 ], 1);
        else if (size <= 256)                            
            position = atomicAdd((int *)&d_bin_counter[8 ], 1);
        else if (size <= 512)                            
            position = atomicAdd((int *)&d_bin_counter[9 ], 1);
        else if (size <= 1024)
            position = atomicAdd((int *)&d_bin_counter[10], 1);
        else if (size <= 2048)
            position = atomicAdd((int *)&d_bin_counter[11], 1);
        else
            position = atomicAdd((int *)&d_bin_counter[12], 1);
        d_bin_segs_id[position] = gid;
    }
}
/*************************** BB_BIN.H (end)  ***************************/

/************************* BB_SEGSORT.H (start)  ************************/
int bb_segsort(int *keys_d, int *vals_d, int n,  int *d_segs, int length)
{
    cudaError_t cuda_err;
    int *h_bin_counter = new int[SEGBIN_NUM];

    int *d_bin_counter;
    int *d_bin_segs_id;
    cuda_err = cudaMalloc((void **)&d_bin_counter, SEGBIN_NUM * sizeof(int));
    ERR_INFO(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, length * sizeof(int));
    ERR_INFO(cuda_err, "alloc d_bin_segs_id");

    cuda_err = cudaMemset(d_bin_counter, 0, SEGBIN_NUM * sizeof(int));
    ERR_INFO(cuda_err, "memset d_bin_counter");

    int *keysB_d;
    int *valsB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, n * sizeof(int));
    ERR_INFO(cuda_err, "alloc keysB_d");
    cuda_err = cudaMalloc((void **)&valsB_d, n * sizeof(int));
    ERR_INFO(cuda_err, "alloc valsB_d");

    bb_bin(d_bin_segs_id, d_bin_counter, d_segs, length, n, h_bin_counter);

    cudaStream_t streams[SEGBIN_NUM-1];
    for(int i = 0; i < SEGBIN_NUM-1; i++) cudaStreamCreate(&streams[i]);

    int subwarp_size, subwarp_num, factor;
    dim3 blocks(256, 1, 1);
    dim3 grids(1, 1, 1);

    blocks.x = 256;
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];
    grids.x = (subwarp_num+blocks.x-1)/blocks.x;
    if(subwarp_num > 0)
    gen_copy<<<grids, blocks, 0, streams[0]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[0], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp2_tc1_r2_r2_orig<<<grids, blocks, 0, streams[1]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[1], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc2_r3_r4_orig<<<grids, blocks, 0, streams[2]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[2], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc4_r5_r8_orig<<<grids, blocks, 0, streams[3]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[3], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 4;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp4_tc4_r9_r16_strd<<<grids, blocks, 0, streams[4]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[4], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp8_tc4_r17_r32_strd<<<grids, blocks, 0, streams[5]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[5], subwarp_num, length);

    blocks.x = 128;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp16_tc4_r33_r64_strd<<<grids, blocks, 0, streams[6]>>>(keys_d, vals_d, keysB_d, valsB_d, 
        n, d_segs, d_bin_segs_id+h_bin_counter[6], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp8_tc16_r65_r128_strd<<<grids, blocks, 0, streams[7]>>>(keys_d, vals_d, keysB_d, valsB_d,  
        n, d_segs, d_bin_segs_id+h_bin_counter[7], subwarp_num, length);

    blocks.x = 256;
    subwarp_size = 32;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    factor = blocks.x/subwarp_size;
    grids.x = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp32_tc8_r129_r256_strd<<<grids, blocks, 0, streams[8]>>>(keys_d, vals_d, keysB_d, valsB_d,  
        n, d_segs, d_bin_segs_id+h_bin_counter[8], subwarp_num, length);

    blocks.x = 128;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk128_tc4_r257_r512_orig<<<grids, blocks, 0, streams[9]>>>(keys_d, vals_d, keysB_d, valsB_d,   
        n, d_segs, d_bin_segs_id+h_bin_counter[9], subwarp_num, length);

    blocks.x = 256;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk256_tc4_r513_r1024_orig<<<grids, blocks, 0, streams[10]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[10], subwarp_num, length);

    blocks.x = 512;
    subwarp_num = h_bin_counter[12]-h_bin_counter[11];
    grids.x = subwarp_num;
    if(subwarp_num > 0)
    gen_bk512_tc4_r1025_r2048_orig<<<grids, blocks, 0, streams[11]>>>(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[11], subwarp_num, length);

    // sort long segments
    subwarp_num = length-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(keys_d, vals_d, keysB_d, valsB_d,
        n, d_segs, d_bin_segs_id+h_bin_counter[12], subwarp_num, length);
    
    // std::swap(keys_d, keysB_d);
    // std::swap(vals_d, valsB_d);
    cuda_err = cudaMemcpy(keys_d, keysB_d, sizeof(int)*n, cudaMemcpyDeviceToDevice);
    ERR_INFO(cuda_err, "copy to keys_d from keysB_d");
    cuda_err = cudaMemcpy(vals_d, valsB_d, sizeof(int)*n, cudaMemcpyDeviceToDevice);
    ERR_INFO(cuda_err, "copy to vals_d from valsB_d");

    cuda_err = cudaFree(d_bin_counter);
    ERR_INFO(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    ERR_INFO(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    ERR_INFO(cuda_err, "free keysB");
    cuda_err = cudaFree(valsB_d);
    ERR_INFO(cuda_err, "free valsB");

    delete[] h_bin_counter;
    return 1;
}

void show_d(int *arr_d, int n, std::string prompt)
{
    std::vector<int> arr_h(n);
    cudaMemcpy(&arr_h[0], arr_d, sizeof(int)*n, cudaMemcpyDeviceToHost);
    std::cout << prompt;
    for(auto v: arr_h) std::cout << v << ", "; std::cout << std::endl;
}
/*************************** BB_SEGSORT.H (end)  ************************/

/*************************** MAIN.H (start)  *************************/
__global__
void spmv_kernel128_sparse_v2l(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
    int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
    int lane = (threadIdx.x&(MFACTOR-1));
    int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
    int offset2 = offset + MFACTOR;
    int offset3 = offset + MFACTOR*2;
    int offset4 = offset + MFACTOR*3;
    int i, j;


    //FTYPE r, r2;
    FTYPE o1 = vout[idx*sc+offset];
    FTYPE o2 = vout[idx*sc+offset2];
    FTYPE o3 = vout[idx*sc+offset3];
    FTYPE o4 = vout[idx*sc+offset4];

    int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);
    int loc1 = mcsr_e[dummy-1], loc2 = mcsr_e[dummy];

    loc1 += ((loc2 - loc1)/STHRESHOLD)*STHRESHOLD;

    int buf; //FTYPE buf2;
    int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
    int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

    int jj=0, l;
    FTYPE r0=0.0f;

    buf = csr_e[loc1+lane];
    for(l=loc1; l<interm2; l+=4) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        int e2 = __shfl(buf, jj+1,MFACTOR)*sc;
        int e3 = __shfl(buf, jj+2,MFACTOR)*sc;
        int e4 = __shfl(buf, jj+3,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;
        FTYPE r2 = vin[e2+offset] * o1;
        r2 += vin[e2+offset2] * o2;
        r2 += vin[e2+offset3] * o3;
        r2 += vin[e2+offset4] * o4;
        FTYPE r3 = vin[e3+offset] * o1;
        r3 += vin[e3+offset2] * o2;
        r3 += vin[e3+offset3] * o3;
        r3 += vin[e3+offset4] * o4;
        FTYPE r4 = vin[e4+offset] * o1;
        r4 += vin[e4+offset2] * o2;
        r4 += vin[e4+offset3] * o3;
        r4 += vin[e4+offset4] * o4;

        int lv = (lane & 3);
        if(lv == 1) {
            FTYPE t = r2; r2 = r1; r1 = t;
            t = r4; r4 = r3; r3 = t;    
        } else if(lv == 2){
            FTYPE t = r3; r3 = r1; r1 = t;
            t = r4; r4 = r2; r2 = t;    
        } else if(lv == 3){
            FTYPE t = r4; r4 = r1; r1 = t;
            t = r3; r3 = r2; r2 = t;    
        } 

        r2 = __shfl_xor(r2, 1);
        r3 = __shfl_xor(r3, 2);
        r4 = __shfl_xor(r4, 3);
        r1 += r2+r3+r4;
        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);

        if(lane >= jj && lane < jj+4) {
            r0 = r1;
        }
        jj = ((jj+4)&(MFACTOR-1));
        if(jj == 0) {
            buf = csr_e[l+4+lane];
            //atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
            ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
        }
    }
    if(interm2 < interm3) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        int e2 = __shfl(buf, jj+1,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;
        FTYPE r2 = vin[e2+offset] * o1;
        r2 += vin[e2+offset2] * o2;
        r2 += vin[e2+offset3] * o3;
        r2 += vin[e2+offset4] * o4;

        int lv = (lane & 1);
        if(lv == 1) {
            FTYPE t = r2; r2 = r1; r1 = t;
        }

        r2 = __shfl_xor(r2, 1);
        r1 += r2;
        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);
        r1 += __shfl_xor(r1, 2);

        if(lane >= jj && lane < jj+2) {
            r0 = r1;
        }

        jj = (jj+2);
    }
    if(interm3 < loc2) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;

        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);
        r1 += __shfl_xor(r1, 2);
        r1 += __shfl_xor(r1, 1);

        if(lane == jj) {
            r0 = r1;
        }

        jj = (jj+1);
    }
    l = loc1+(((l - loc1)>>5)<<5);
    if(lane < jj) {
        //atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
        ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
    }
}   

__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel128_sparse_v2h(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev, int *special, int *special2)
{
    int idx = special[blockIdx.x];// + (threadIdx.x>>(LOG_MFACTOR));
    int lane = (threadIdx.x&(MFACTOR-1));
    int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
    int offset2 = offset + MFACTOR;
    int offset3 = offset + MFACTOR*2;
    int offset4 = offset + MFACTOR*3;
    int i, j;

    FTYPE o1 = vout[idx*sc+offset];
    FTYPE o2 = vout[idx*sc+offset2];
    FTYPE o3 = vout[idx*sc+offset3];
    FTYPE o4 = vout[idx*sc+offset4];

    int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);

    int loc1 = mcsr_e[dummy-1] + special2[blockIdx.x] + ((threadIdx.x>>5)*SSTRIDE);

    int buf; //FTYPE buf2;
         //int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
         //int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

    int jj=0, l;
    FTYPE r0=0.0f;

    buf = csr_e[loc1+lane];
    for(l=loc1; l<loc1+SSTRIDE; l+=4) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        int e2 = __shfl(buf, jj+1,MFACTOR)*sc;
        int e3 = __shfl(buf, jj+2,MFACTOR)*sc;
        int e4 = __shfl(buf, jj+3,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;
        FTYPE r2 = vin[e2+offset] * o1;
        r2 += vin[e2+offset2] * o2;
        r2 += vin[e2+offset3] * o3;
        r2 += vin[e2+offset4] * o4;
        FTYPE r3 = vin[e3+offset] * o1;
        r3 += vin[e3+offset2] * o2;
        r3 += vin[e3+offset3] * o3;
        r3 += vin[e3+offset4] * o4;
        FTYPE r4 = vin[e4+offset] * o1;
        r4 += vin[e4+offset2] * o2;
        r4 += vin[e4+offset3] * o3;
        r4 += vin[e4+offset4] * o4;

        int lv = (lane & 3);
        if(lv == 1) {
            FTYPE t = r2; r2 = r1; r1 = t;
            t = r4; r4 = r3; r3 = t;    
        } else if(lv == 2){
            FTYPE t = r3; r3 = r1; r1 = t;
            t = r4; r4 = r2; r2 = t;    
        } else if(lv == 3){
            FTYPE t = r4; r4 = r1; r1 = t;
            t = r3; r3 = r2; r2 = t;    
        } 

        r2 = __shfl_xor(r2, 1);
        r3 = __shfl_xor(r3, 2);
        r4 = __shfl_xor(r4, 3);
        r1 += r2+r3+r4;
        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);

        if(lane >= jj && lane < jj+4) {
            r0 = r1;
        }
        jj = ((jj+4)&(MFACTOR-1));
        if(jj == 0) {
            buf = csr_e[l+4+lane];
            //atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
            ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
        }
    }

    //if(threadIdx.x == 0) printf("%d %f %f\n", blockIdx.x, r, r2);
}

    __global__
    __launch_bounds__(BSIZE, 2048/BSIZE)

__global__
void spmv_kernel128_sparse_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
    int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
    int lane = (threadIdx.x&(MFACTOR-1));
    int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
    int offset2 = offset + MFACTOR;
    int offset3 = offset + MFACTOR*2;
    int offset4 = offset + MFACTOR*3;
    int i, j;


    //FTYPE r, r2;
    FTYPE o1 = vout[idx*sc+offset];
    FTYPE o2 = vout[idx*sc+offset2];
    FTYPE o3 = vout[idx*sc+offset3];
    FTYPE o4 = vout[idx*sc+offset4];
    int dummy = mcsr_cnt[idx/BH]*BH + ((idx&(BH-1))+1)*(mcsr_cnt[idx/BH+1] - mcsr_cnt[idx/BH]);
    int loc1 = mcsr_e[dummy-1], loc2 = mcsr_e[dummy];

    int buf; //FTYPE buf2;
    int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
    int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

    int jj=0, l;
    FTYPE r0=0.0f;

    buf = csr_e[loc1+lane];
    for(l=loc1; l<interm2; l+=4) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        int e2 = __shfl(buf, jj+1,MFACTOR)*sc;
        int e3 = __shfl(buf, jj+2,MFACTOR)*sc;
        int e4 = __shfl(buf, jj+3,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;
        FTYPE r2 = vin[e2+offset] * o1;
        r2 += vin[e2+offset2] * o2;
        r2 += vin[e2+offset3] * o3;
        r2 += vin[e2+offset4] * o4;
        FTYPE r3 = vin[e3+offset] * o1;
        r3 += vin[e3+offset2] * o2;
        r3 += vin[e3+offset3] * o3;
        r3 += vin[e3+offset4] * o4;
        FTYPE r4 = vin[e4+offset] * o1;
        r4 += vin[e4+offset2] * o2;
        r4 += vin[e4+offset3] * o3;
        r4 += vin[e4+offset4] * o4;

        int lv = (lane & 3);
        if(lv == 1) {
            FTYPE t = r2; r2 = r1; r1 = t;
            t = r4; r4 = r3; r3 = t;    
        } else if(lv == 2){
            FTYPE t = r3; r3 = r1; r1 = t;
            t = r4; r4 = r2; r2 = t;    
        } else if(lv == 3){
            FTYPE t = r4; r4 = r1; r1 = t;
            t = r3; r3 = r2; r2 = t;    
        } 

        r2 = __shfl_xor(r2, 1);
        r3 = __shfl_xor(r3, 2);
        r4 = __shfl_xor(r4, 3);
        r1 += r2+r3+r4;
        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);

        if(lane >= jj && lane < jj+4) {
            r0 = r1;
        }
        jj = ((jj+4)&(MFACTOR-1));
        if(jj == 0) {
            buf = csr_e[l+4+lane];
            //atomicAdd(&ocsr_ev[l-28+lane], r0 * csr_ev[l-28+lane]);
            ocsr_ev[l-28+lane] = r0 * csr_ev[l-28+lane];
        }
    }
    if(interm2 < interm3) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        int e2 = __shfl(buf, jj+1,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;
        FTYPE r2 = vin[e2+offset] * o1;
        r2 += vin[e2+offset2] * o2;
        r2 += vin[e2+offset3] * o3;
        r2 += vin[e2+offset4] * o4;

        int lv = (lane & 1);
        if(lv == 1) {
            FTYPE t = r2; r2 = r1; r1 = t;
        }

        r2 = __shfl_xor(r2, 1);
        r1 += r2;
        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);
        r1 += __shfl_xor(r1, 2);

        if(lane >= jj && lane < jj+2) {
            r0 = r1;
        }

        jj = (jj+2);
    }
    if(interm3 < loc2) {
        int e1 = __shfl(buf, jj,MFACTOR)*sc;
        FTYPE r1 = vin[e1+offset] * o1;
        r1 += vin[e1+offset2] * o2;
        r1 += vin[e1+offset3] * o3;
        r1 += vin[e1+offset4] * o4;

        r1 += __shfl_xor(r1, 16);
        r1 += __shfl_xor(r1, 8);
        r1 += __shfl_xor(r1, 4);
        r1 += __shfl_xor(r1, 2);
        r1 += __shfl_xor(r1, 1);

        if(lane == jj) {
            r0 = r1;
        }

        jj = (jj+1);
    }
    l = loc1+(((l - loc1)>>5)<<5);
    if(lane < jj) {
        //atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
        ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
    }
}   

__global__
//__launch_bounds__(BSIZE, 2048/BSIZE)
void spmv_kernel64_ssparse(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, FTYPE *ocsr_ev)
{
    int idx = (blockIdx.x*SBF)+(threadIdx.x>>5);// + (threadIdx.x>>(LOG_MFACTOR));
    int lane = (threadIdx.x&(MFACTOR-1));
    int offset = (blockIdx.z<<(LOG_MFACTOR+1))+lane;
    int offset2 = offset + MFACTOR;
    int i, j;

    FTYPE r;//=0.0f;
    FTYPE r2;//=0.0f;
    int loc1 = csr_v[idx], loc2 = csr_v[idx+1];

    int buf; FTYPE buf2;
    int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

    int jj=0, l;

    for(l=loc1; l<loc2; l++) {
        r = vin[csr_e[l]*sc+offset] * vout[idx*sc+offset];
        r2 = vin[csr_e[l]*sc+offset2] * vout[idx*sc+offset2];
        r += r2;
        r += __shfl_down(r, 16);
        r += __shfl_down(r, 8);
        r += __shfl_down(r, 4);
        r += __shfl_down(r, 2);
        r += __shfl_down(r, 1);
        //printf("%f\n", r);
        if(lane == 0) {
            //atomicAdd(&ocsr_ev[l], r*csr_ev[l]); // can be avoided
            atomicAdd(&ocsr_ev[l], r*csr_ev[l]); // can be avoided
        }
    }
}

__global__
void spmv_kernel128_dense_v2(int sc, int *csr_v, int *csr_e, FTYPE *csr_ev, int *mcsr_cnt, int *mcsr_e, int *mcsr_list, FTYPE *vin, FTYPE *vout, int *baddr, int *saddr, FTYPE *ocsr_ev)
{
    int lane = (threadIdx.x&(MFACTOR-1));
    int offset = (blockIdx.z<<(LOG_MFACTOR+2))+lane;
    int offset2 = offset + MFACTOR;
    int offset3 = offset + MFACTOR*2;
    int offset4 = offset + MFACTOR*3;
    int loop, i, j;

    __shared__ FTYPE sin[BW][MFACTOR];
    __shared__ FTYPE sin2[BW][MFACTOR];
    __shared__ FTYPE sin3[BW][MFACTOR];
    __shared__ FTYPE sin4[BW][MFACTOR];

    int warp_id = (threadIdx.x>>LOG_MFACTOR);

    int base_addr = baddr[blockIdx.x];
    int stride = saddr[blockIdx.x];

    for(i=warp_id;i<BW;i+=(DBSIZE>>LOG_MFACTOR)) {
        int hash = mcsr_list[blockIdx.x*BW + i];
        if(hash >= 0) {
            sin[hash%BW][lane] = vin[hash*sc + offset];
            sin2[hash%BW][lane] = vin[hash*sc + offset2];
            sin3[hash%BW][lane] = vin[hash*sc + offset3];
            sin4[hash%BW][lane] = vin[hash*sc + offset4];
        }
    }
    __syncthreads();

    for(i=warp_id;i<BH;i+=(DBSIZE>>LOG_MFACTOR)) {

        FTYPE o1 = vout[(base_addr*BH+i)*sc+offset];
        FTYPE o2 = vout[(base_addr*BH+i)*sc+offset2];
        FTYPE o3 = vout[(base_addr*BH+i)*sc+offset3];
        FTYPE o4 = vout[(base_addr*BH+i)*sc+offset4];


        int dummy = mcsr_cnt[base_addr]*BH + i*(mcsr_cnt[base_addr+1] - mcsr_cnt[base_addr]) + stride;
        int loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

        int buf; //FTYPE buf2;
             //int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
        int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

        int jj=0, l;
        FTYPE r0=0.0f;

        buf = csr_e[loc1+lane];
        for(l=loc1; l<interm3; l+=2) {
            int e1 = (__shfl(buf, jj,MFACTOR)&(BW-1));
            int e2 = (__shfl(buf, jj+1,MFACTOR)&(BW-1));

            FTYPE r1 = sin[e1][lane] * o1;
            r1 += sin2[e1][lane] * o2;
            r1 += sin3[e1][lane] * o3;
            r1 += sin4[e1][lane] * o4;
            FTYPE r2 = sin[e2][lane] * o1;
            r2 += sin2[e2][lane] * o2;
            r2 += sin3[e2][lane] * o3;
            r2 += sin4[e2][lane] * o4;

            int lv = (lane & 1);
            if(lv == 1) {
                FTYPE t = r2; r2 = r1; r1 = t;
            }

            r2 = __shfl_xor(r2, 1);
            r1 += r2;
            //if(threadIdx.x == 0) printf("b: %f ", r1);
            r1 += __shfl_xor(r1, 16);
            r1 += __shfl_xor(r1, 8);
            r1 += __shfl_xor(r1, 4);
            r1 += __shfl_xor(r1, 2);
            //if(threadIdx.x == 0) printf("a: %f ", r1);

            if(lane >= jj && lane < jj+2) {
                r0 = r1;
            }

            jj = ((jj+2)&(MFACTOR-1));
            if(jj == 0) {
                buf = csr_e[l+2+lane];
                //atomicAdd(&ocsr_ev[l-30+lane], r0 * csr_ev[l-30+lane]);
                ocsr_ev[l-30+lane] = r0 * csr_ev[l-30+lane];
                //if(l-30+lane <32) printf("(%d: %f %f %f %f)", l-30+lane, ocsr_ev[l-30+lane], r0*csr_ev[l-30+lane], r0, csr_ev[l-30+lane]);
            }
        }


        if(interm3 < loc2) {
            int e1 = (__shfl(buf, jj,MFACTOR)&(BW-1));

            FTYPE r1 = sin[e1][lane] * o1;
            r1 += sin2[e1][lane] * o2;
            r1 += sin3[e1][lane] * o3;
            r1 += sin4[e1][lane] * o4;
            //r1 += vin[e1+offset3] * o3;
            //r1 += vin[e1+offset4] * o4;

            r1 += __shfl_xor(r1, 16);
            r1 += __shfl_xor(r1, 8);
            r1 += __shfl_xor(r1, 4);
            r1 += __shfl_xor(r1, 2);
            r1 += __shfl_xor(r1, 1);

            if(lane == jj) {
                r0 = r1;
            }

            jj = (jj+1);
        }
        l = loc1+(((l - loc1)>>5)<<5);
        if(lane < jj) {
            //atomicAdd(&ocsr_ev[l+lane], r0 * csr_ev[l+lane]);
            ocsr_ev[l+lane] = r0 * csr_ev[l+lane];
        }



    }
}

__global__ void dense_block_detect(int *csr_v, int *mcsr_chk, int *csr_e0, int *flag)
{
	int i;
	int lb = csr_v[blockIdx.x*BH];
	int ub = csr_v[(blockIdx.x+1)*BH];
	//__shared__ short scr_pad[SC_SIZE];
	__shared__ int scr_pad[SC_SIZE];

	for(i=threadIdx.x; i<SC_SIZE; i+=blockDim.x) {
		scr_pad[i] = 0;
	}
	__syncthreads();
	for(i=lb+threadIdx.x; i<ub; i+=blockDim.x) {
		int key = (csr_e0[i]&(SC_SIZE-1));
		if(scr_pad[key] < THRESHOLD) atomicAdd(&scr_pad[key], 1);
	}
	__syncthreads();
	int r=0;
	for(i=threadIdx.x; i<SC_SIZE; i+=blockDim.x) {
		if(scr_pad[i] >= THRESHOLD) r++;
	}
	__syncthreads();
	r += __shfl_down(r, 16);
	r += __shfl_down(r, 8);
	r += __shfl_down(r, 4);
	r += __shfl_down(r, 2);
	r += __shfl_down(r, 1);
	if((threadIdx.x & 31) == 0) scr_pad[threadIdx.x>>5] = r;
	__syncthreads();
	if(threadIdx.x == 0) {
		for(i=1; i<BH/32; i++)
			r += scr_pad[i];
		if(r >= MIN_OCC) {
			mcsr_chk[blockIdx.x] = 1;
			if(flag[blockIdx.x&127] == 0) flag[blockIdx.x&127] = 1;
		}
	}
}

__global__ void simple_mcsr_cnt(int npanel, int *mcsr_cnt)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < npanel) mcsr_cnt[idx] = idx;
}

__global__ void csr_pivot_gen(int npanel, int *csr_v, int *csr_pivot)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < npanel) {
		csr_pivot[idx] = csr_v[(idx)*BH];
	}
}

__global__ void csr_pnt_gen(int ne, int *csr_e0, int *key, STYPE *key2, int *val)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < ne) {
		key[idx] = csr_e0[idx];
		key2[idx] = 30000; 
		val[idx] = idx;
	}
}

__global__ void mcsr_cnt_calc(int *csr_pivot, int *key, int *mcsr_cnt, int *mcsr_chk)
{
	if(mcsr_chk[blockIdx.x] == 0) return;
	int lb = csr_pivot[blockIdx.x]+THRESHOLD-1;
	int ub = csr_pivot[blockIdx.x+1];
	__shared__ int age[BW];
	__shared__ int occ[MCSR_CNT_SIZE];
	for(int i=threadIdx.x; i<BW; i+=blockDim.x) {
		age[i] = 0;
	}
	for(int i=threadIdx.x; i<MCSR_CNT_SIZE; i+=blockDim.x) {
		if(i > 0) occ[i] = 0;
		else occ[i] = BW;
	}
	__syncthreads();

	for(int i=lb+threadIdx.x; i<ub; i+=blockDim.x) {
		if(i == ub - 1 || key[i] != key[i+1]) {
			if(key[i] == key[i-(THRESHOLD-1)]) {
				int hash = atomicAdd(&age[key[i]&(BW-1)], 1);
				atomicAdd(&occ[hash+1], 1);
			}
		}
	}
	__syncthreads();
	if(threadIdx.x < MCSR_CNT_SIZE-1 && occ[threadIdx.x] >= MIN_OCC && occ[threadIdx.x+1] < MIN_OCC) {
		mcsr_cnt[blockIdx.x+1] = threadIdx.x;
	} 
}

__global__ void key2_marking(int *csr_pivot, int *key, STYPE *key2, int *val, int *mcsr_cnt, int *mcsr_list, int *baddr, int *saddr, int *mcsr_chk)
{
	if(mcsr_chk[blockIdx.x] == 0) return;
	int lb = csr_pivot[blockIdx.x]+THRESHOLD-1;
	int ub = csr_pivot[blockIdx.x+1];
	int uub = lb+CEIL(ub-lb,1024)*1024;
	int bloc = (mcsr_cnt[blockIdx.x] - blockIdx.x)*BW;
	int limit = mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x] - 1;

	__shared__ int age[BW];
	__shared__ int list[LIST_CANDI];  
	__shared__ short list2[LIST_CANDI];
	__shared__ int listp;
	for(int i=threadIdx.x; i<BW; i+=blockDim.x) {
		age[i] = 0;
	}
	__syncthreads();
	for(int i=threadIdx.x; i<limit; i+=blockDim.x) {
		baddr[mcsr_cnt[blockIdx.x]-blockIdx.x+i] = blockIdx.x;
		saddr[mcsr_cnt[blockIdx.x]-blockIdx.x+i] = threadIdx.x;
	}
	for(int i0=lb+threadIdx.x; i0<uub; i0+=LIST_CANDI*THRESHOLD) {
		if(threadIdx.x == 0) listp=0;
		__syncthreads();
		for(int i=i0; i<MIN(i0+LIST_CANDI*THRESHOLD,ub); i+=blockDim.x) {
			if(i == ub - 1 || key[i] != key[i+1]) {
				if(key[i] == key[i-(THRESHOLD-1)]) {
					int width = (key[i]&(BW-1));
					int depth = atomicAdd(&age[width], 1);
					if(depth < limit) {
						mcsr_list[bloc + depth*BW + width] = key[i];
						int p = atomicAdd(&listp, 1);
						list[p] = i;
						list2[p] = depth;		
					} 			

				}
			}
		}
		__syncthreads();
		#define LLF (8)
		#define LOG_LLF (3)
		for(int i=(threadIdx.x>>LOG_LLF); i<listp; i+=(blockDim.x>>LOG_LLF)) {
			int p = list[i];
			int depth = list2[i];
			int width = (key[p]&(BW-1));
			for(int k=p-(threadIdx.x&(LLF-1)); k >= csr_pivot[blockIdx.x] && key[k] == key[p]; k-=LLF) {
				key2[val[k]] = depth;
			} 	
		}
		__syncthreads();
	}
}

__global__ void fill_val(int ne, int *val)
{
	int idx = blockIdx.x*blockDim.x*4 + threadIdx.x;
	int idx2 = idx + blockDim.x;
	int idx3 = idx + blockDim.x*2;
	int idx4 = idx + blockDim.x*3;
	if(idx4 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
		val[idx3] = idx3;
		val[idx4] = idx4;
	} else if(idx3 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
		val[idx3] = idx3;
	} else if(idx2 < ne) {
		val[idx] = idx;
		val[idx2] = idx2;
	} else if(idx < ne) {
		val[idx] = idx;
	}
}

__global__ void fill_mcsre(int *csr_v, int *mcsr_cnt, STYPE *key2, int *mcsr_e, int *rmv)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int delta = mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x];
	int bidx = mcsr_cnt[blockIdx.x]*BH + delta*threadIdx.x;
	int i = csr_v[idx];
	//int lb, ub=key2[i];
	int kk = MIN(key2[i], delta-1);
	if(i == csr_v[idx+1]) kk = delta-1;
	for(int j = 0; j<=kk; j++) 
		mcsr_e[bidx+j] = csr_v[idx];
	for(; i<csr_v[idx+1]; i++) {
		int lb = key2[i], ub = key2[i+1];
		//lb = ub; ub = key2[i+1];
		if(lb == 30000) break;
		if(i == csr_v[idx+1]-1 || ub >= delta) ub = delta-1;
		for(int j = lb+1; j<=ub; j++) {
			mcsr_e[bidx+j] = i+1;
		}

	}
	//if(bidx >= 18700 && bidx <= 18750) {
	//printf("(%d: %d %d %d)\n", idx, delta, csr_v[idx], csr_v[idx+1]);
	//for(i=0;i<=delta-1;i++) printf("(%d %d: %d %d %d)", idx, i,  mcsr_e[bidx+i], key2[i], delta); printf("\n");
	//}
	int r = (csr_v[idx+1] - mcsr_e[bidx+delta-1]);
	r += __shfl_down(r, 16);
	r += __shfl_down(r, 8);
	r += __shfl_down(r, 4);
	r += __shfl_down(r, 2);
	r += __shfl_down(r, 1);
	if((threadIdx.x&31) == 0) atomicAdd(&rmv[(idx>>5)&127], r);
}

__global__ void porting(int ne, int *val, int *csr_e0, FTYPE *csr_ev0, int *csr_e, FTYPE *csr_ev)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < ne) {
		int k = val[idx];
		csr_e[idx] = csr_e0[k];
		csr_ev[idx] = csr_ev0[k];	
	}
}

__global__ void cal_vari(int nr, double avg, int *mcsr_cnt, int *mcsr_e, double *vari, int *special_bb)
{
	int idx = (mcsr_cnt[blockIdx.x]*BH) + (mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x])*(threadIdx.x+1);
	int i2 = mcsr_e[idx] - mcsr_e[idx-1];
	double r = ((double)i2 - avg);
	double r2 = r*r;

	r2 += __shfl_down(r2, 16);
	r2 += __shfl_down(r2, 8);
	r2 += __shfl_down(r2, 4);
	r2 += __shfl_down(r2, 2);
	r2 += __shfl_down(r2, 1);
	i2 /= STHRESHOLD;
	//if(i2 > 1000) printf("ERR : %d %d %d\n", idx, mcsr_e[idx-1], mcsr_e[idx]);
	i2 += __shfl_down(i2, 16);	
	i2 += __shfl_down(i2, 8);	
	i2 += __shfl_down(i2, 4);	
	i2 += __shfl_down(i2, 2);	
	i2 += __shfl_down(i2, 1);	
	if((threadIdx.x&31) == 0) {
		atomicAdd(&vari[((blockIdx.x*blockDim.x+threadIdx.x)>>5)&127], r2);
		if(i2 > 0) atomicAdd(&special_bb[((blockIdx.x*blockDim.x+threadIdx.x)>>5)&127], i2);
	}
}

__global__ void make_special(int *mcsr_cnt, int *mcsr_e, int *special, int *special2, int *scnt)
{
	int idx = (mcsr_cnt[blockIdx.x]*BH) + (mcsr_cnt[blockIdx.x+1] - mcsr_cnt[blockIdx.x])*(threadIdx.x+1);
	int i2 = (mcsr_e[idx] - mcsr_e[idx-1])/STHRESHOLD;
	if(i2 > 0) {
		int k = atomicAdd(&scnt[0], i2);
		for(int i=k;i<k+i2;i++) {
			special[i] = blockIdx.x*blockDim.x+threadIdx.x;
			special2[i] = STHRESHOLD*(i-k);
		}
	}
}
/**************************  MAIN.H (end)  **************************/
/************************************************************************/
