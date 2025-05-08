#!/bin/bash

path_validation='/various/pmpakos/SpMV-Research/validation_matrices'

cores='24'
max_cores=96
export OMP_NUM_THREADS="$cores"
# export AOCLSPARSE_NUM_THREADS="$cores"
export GOMP_CPU_AFFINITY="0-$((max_cores-1))"

# Encourages idle threads to spin rather than sleep.
export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'

# export AOCL_PATH=/various/pmpakos/spmv_paper/aocl-sparse/build/release/
export AOCL_ROOT=/various/pmpakos/epyc5_libs/aocl-5.0
# export AOCL_PATH=${AOCL_ROOT}/aocl-sparse/build/release/
export AOCL_PATH=${AOCL_ROOT}/aocl-sparse-dev/build/release/

# export MKL_PATH=/various/pmpakos/intel/oneapi/mkl/2024.1/
export MKL_PATH=/various/common_tools/intel_parallel_studio/compilers_and_libraries/linux/mkl
export CUDA_PATH=/usr/local/cuda/

export LD_LIBRARY_PATH=${MKL_PATH}/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${AOCL_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/pmpakos/epyc5_libs/openssl-1.1.1o/:${AOCL_ROOT}/amd-blis/lib/LP64:${AOCL_ROOT}/amd-libflame/lib/LP64:${AOCL_ROOT}/amd-utils/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/pmpakos/sparse_survey/code/sputnik/build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/pmpakos/sparse_survey/code/libtorch_cuda/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/pmpakos/sparse_survey/code/libtorch_cuda/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64/:$LD_LIBRARY_PATH


# lscpu | grep -q -i amd
# if (($? == 0)); then
#     export MKL_DEBUG_CPU_TYPE=5
# fi
# export MKL_ENABLE_INSTRUCTIONS=AVX512
# export MKL_VERBOSE=1

matrices_validation=(
    scircuit.mtx
    # mac_econ_fwd500.mtx
    # raefsky3.mtx
    # rgg_n_2_17_s0.mtx
    # bbmat.mtx
    # appu.mtx
    # mc2depi.mtx
    # rma10.mtx
    # cop20k_A.mtx
    # thermomech_dK.mtx
    # webbase-1M.mtx
    # cant.mtx
    # ASIC_680k.mtx
    # roadNet-TX.mtx
    # pdb1HYS.mtx
    # TSOPF_RS_b300_c3.mtx
    # Chebyshev4.mtx
    # consph.mtx
    # com-Youtube.mtx
    # rajat30.mtx
    # radiation.mtx
    # Stanford_Berkeley.mtx
    # shipsec1.mtx
    # PR02R.mtx
    # CurlCurl_2.mtx
    # gupta3.mtx
    # mip1.mtx
    # rail4284.mtx
    # pwtk.mtx
    # crankseg_2.mtx
    # Si41Ge41H72.mtx
    # TSOPF_RS_b2383.mtx
    # in-2004.mtx
    # Ga41As41H72.mtx
)

matrices=(
    "${matrices_validation[@]}"
)


make clean;
time make -j
# time make
# export OMP_DISPLAY_ENV=TRUE

for a in "${matrices[@]}"
do
    echo '--------'
    echo ${path_validation}/$a
    # ./mat_read.exe ${path_validation}/$a
    ./mat_cusparse_spmm.exe ${path_validation}/$a 128
    # ./mat_gnnpilot_spmm.exe ${path_validation}/$a 128 1
    # ./mat_gnnpilot_spmm.exe ${path_validation}/$a 128 2
    # ./mat_gnnpilot_sddmm.exe ${path_validation}/$a 128 1
    # ./mat_gnnpilot_sddmm.exe ${path_validation}/$a 128 2
done

# for a in "${matrices[@]}"
# do
    # echo '--------'
    # echo ${path_validation}/$a
    # ./mat_read.exe ${path_validation}/$a

    ####################### GPU #######################
    # cuSPARSE
    # export CUSPARSE_LOG_LEVEL=5
    # ./mat_cusparse_spmv.exe ${path_validation}/$a
    # for k in 128;
    # do
    #     ./mat_cusparse_spmm.exe ${path_validation}/$a ${k}
    # done
    # for middle in 16;
    # do
    #     ./mat_cusparse_sddmm.exe ${path_validation}/$a ${middle}
    # done

    # dgSPARSE
    # for k in 128;
    # do
    #     # 0: GESPMM_ALG_SEQREDUCE_ROWBALANCE
    #     # 1: GESPMM_ALG_PARREDUCE_ROWBALANCE
    #     # 2: GESPMM_ALG_SEQREDUCE_NNZBALANCE
    #     # 3: GESPMM_ALG_PARREDUCE_NNZBALANCE
    #     # 4: GESPMM_ALG_ROWCACHING_ROWBALANCE
    #     # 5: GESPMM_ALG_ROWCACHING_NNZBALANCE
    #     for method in 0 1 2 3 4 5;
    #     do
    #         ./mat_dgsparse_spmm.exe ${path_validation}/$a ${k} ${method}
    #     done
    # done
    # # for middle in 16 64 256;
    # for middle in 16;
    # do
    #     ./mat_dgsparse_sddmm.exe ${path_validation}/$a ${middle}
    # done

    # SPUTNIK
    # for k in 128;
    # do
    #     ./mat_sputnik_spmm.exe ${path_validation}/$a ${k}
    # done
    # for middle in 16;
    # do
    #     ./mat_sputnik_sddmm.exe ${path_validation}/$a ${middle}
    # done

    # ACC
    # for k in 128;
    # do
    #     ./mat_acc_spmm.exe ${path_validation}/$a ${k}
    # done

    # ASpT
    # for k in 128;
    # do
    #     ./mat_aspt_spmm.exe ${path_validation}/$a ${k}
    # done
    # for middle in 16;
    # do
    #     ./mat_aspt_sddmm.exe ${path_validation}/$a ${middle}
    # done

    # RoDe
    # for k in 128;
    # do
    #     ./mat_rode_spmm.exe ${path_validation}/$a ${k}
    # done
    # for middle in 128;
    # do
    #     ./mat_rode_sddmm.exe ${path_validation}/$a ${middle}
    # done

    # DTC-v1 ("float_nonsplit", "float2_nonsplit", "float2_split", "float4_nonsplit", "float4_split")
    # for k in 128;
    # do
    #     for exeplan in "float_nonsplit" "float2_nonsplit" "float2_split" "float4_nonsplit" "float4_split";
    #     do
    #         ./mat_dtc_v1_spmm.exe ${path_validation}/$a ${k} ${exeplan}
    #     done
    # done
    # DTC-v2 ("float_nonsplit", "float4_split")
    # for k in 128;
    # do
    #     for exeplan in "float_nonsplit" "float4_split";
    #     do
    #         ./mat_dtc_v2_spmm.exe ${path_validation}/$a ${k} ${exeplan}
    #     done
    # done

    # HC-SpMM (works for >128 only)
    # for k in 128;
    # do
    #     ./mat_hc_spmm.exe ${path_validation}/$a ${k}
    # done

    ####################### CPU #######################
    
    # MKL
    # ./mat_mkl_spmv.exe ${path_validation}/$a
    # for k in 16;
    # do
    #     ./mat_mkl_spmm.exe ${path_validation}/$a ${k}
    # done
    
    # AOCL-Sparse
    # ./mat_aocl_spmv.exe ${path_validation}/$a
    # for k in 16;
    # do
    #     ./mat_aocl_spmm.exe ${path_validation}/$a ${k}
    # done

    # export AOCL_PATH=/various/pmpakos/epyc5_libs/aocl-sparse-4.0/build/release/
    # export LD_LIBRARY_PATH="${AOCL_PATH}/lib"
    # ./mat_aocl_spmv4.exe ${path_validation}/$a

    # export AOCL_PATH=/various/pmpakos/epyc5_libs/aocl-sparse-3.2/build/release/
    # export LD_LIBRARY_PATH="${AOCL_PATH}/lib"
    # ./mat_aocl_spmv3.exe ${path_validation}/$a

    # ASpT-CPU
    # for k in 16;
    # do
    #     ./mat_aspt_spmm_cpu.exe ${path_validation}/$a ${k}
    # done
    # for middle in 16;
    # do
    #     ./mat_aspt_sddmm_cpu.exe ${path_validation}/$a ${middle}
    # done

    # FusedMM
    # for k in 16;
    # do
    #     ./mat_fused_spmm.exe ${path_validation}/$a ${k}
    # done

# done
