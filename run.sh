#!/bin/bash

#####################################################################################################
#### No need to edit the following paths! ###########################################################
#####################################################################################################
# export AOCL_PATH=/various/pmpakos/spmv_paper/aocl-sparse/build/release/
export AOCL_ROOT=/various/pmpakos/epyc5_libs/aocl-5.0
# export AOCL_PATH=${AOCL_ROOT}/aocl-sparse/build/release/
export AOCL_PATH=${AOCL_ROOT}/aocl-sparse-dev/build/release

# export MKL_PATH=/various/pmpakos/intel/oneapi/mkl/2024.1/
export MKL_PATH=/various/common_tools/intel_parallel_studio/compilers_and_libraries/linux/mkl
export CUDA_PATH=/usr/local/cuda/

export LD_LIBRARY_PATH=${MKL_PATH}/lib/intel64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${AOCL_PATH}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/pmpakos/epyc5_libs/openssl-1.1.1o/:${AOCL_ROOT}/amd-blis/lib/LP64:${AOCL_ROOT}/amd-libflame/lib/LP64:${AOCL_ROOT}/amd-utils/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/deps/sputnik/build/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/deps/libtorch/lib/:$LD_LIBRARY_PATH
# lscpu | grep -q -i amd
# if (($? == 0)); then
#     export MKL_DEBUG_CPU_TYPE=5
# fi
# export MKL_ENABLE_INSTRUCTIONS=AVX512
# export MKL_VERBOSE=1
#####################################################################################################
#####################################################################################################

# Number of cores used for OpenMP parallelization.
cores='24'
max_cores=96
export OMP_NUM_THREADS="$cores"
# export AOCLSPARSE_NUM_THREADS="$cores"
export GOMP_CPU_AFFINITY="0-$((max_cores-1))"

# Encourages idle threads to spin rather than sleep.
export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'

path_validation='/various/pmpakos/SpMV-Research/validation_matrices'
# Select on which matrices from the following list you want to run the benchmarks.
# These matrices are located on the above defined path.
matrices=(
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

make clean;
time make -j

# For CPU kernels, no need for 1000 extra iterations for warmup, just change the environment variable
export GPU_KERNEL=0
echo "CPU kernels"
for k in 128;
do
    for a in "${matrices[@]}"
    do
        echo '--------'
        echo ${path_validation}/$a
        ./spmm_mkl.exe ${path_validation}/$a $k
        ./spmm_aocl.exe ${path_validation}/$a $k
        ./spmm_aspt_cpu.exe ${path_validation}/$a $k
        ./spmm_fusedmm.exe ${path_validation}/$a $k

        ./sddmm_aspt_cpu.exe ${path_validation}/$a $k
    done
done

# For GPU kernels, we need to run 1000 extra iterations for warmup.
export GPU_KERNEL=1
echo "GPU kernels"
for k in 128;
do
    for a in "${matrices[@]}"
    do
        echo '--------'
        echo ${path_validation}/$a
        ./spmm_cusparse.exe ${path_validation}/$a $k
        ./spmm_acc.exe ${path_validation}/$a $k
        ./spmm_aspt_gpu.exe ${path_validation}/$a $k
        ./spmm_rode.exe ${path_validation}/$a $k
        ./spmm_hc.exe ${path_validation}/$a $k
        ./spmm_dgsparse_0.exe ${path_validation}/$a $k # GESPMM_ALG_SEQREDUCE_ROWBALANCE
        ./spmm_dgsparse_1.exe ${path_validation}/$a $k # GESPMM_ALG_PARREDUCE_ROWBALANCE
        ./spmm_dgsparse_2.exe ${path_validation}/$a $k # GESPMM_ALG_SEQREDUCE_NNZBALANCE
        ./spmm_dgsparse_3.exe ${path_validation}/$a $k # GESPMM_ALG_PARREDUCE_NNZBALANCE
        ./spmm_dgsparse_4.exe ${path_validation}/$a $k # GESPMM_ALG_ROWCACHING_ROWBALANCE
        ./spmm_dgsparse_5.exe ${path_validation}/$a $k # GESPMM_ALG_ROWCACHING_NNZBALANCE
        ./spmm_gnnpilot_1.exe ${path_validation}/$a $k # BALANCE=1
        ./spmm_gnnpilot_2.exe ${path_validation}/$a $k # BALANCE=2
        ./spmm_dtc_0.exe ${path_validation}/$a $k # float_nonsplit
        ./spmm_dtc_1.exe ${path_validation}/$a $k # float2_nonsplit
        ./spmm_dtc_2.exe ${path_validation}/$a $k # float2_split
        ./spmm_dtc_3.exe ${path_validation}/$a $k # float4_nonsplit
        ./spmm_dtc_4.exe ${path_validation}/$a $k # float4_split
        ./spmm_dtc_5.exe ${path_validation}/$a $k # v2 float_nonsplit
        ./spmm_dtc_6.exe ${path_validation}/$a $k # v2 float4_split
        ./spmm_sputnik.exe ${path_validation}/$a $k

        ./sddmm_cusparse.exe ${path_validation}/$a $k
        ./sddmm_aspt_gpu.exe ${path_validation}/$a $k
        ./sddmm_rode.exe ${path_validation}/$a $k
        ./sddmm_dgsparse.exe ${path_validation}/$a $k
        ./sddmm_gnnpilot.exe ${path_validation}/$a $k
        ./sddmm_sputnik.exe ${path_validation}/$a $k
    done
done

