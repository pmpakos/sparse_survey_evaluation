# sparse_survey_evaluation

This is the repo containing the code for the pure computational kernels for SpMM (Sparse Matrix-Matrix Multiplication) and SDDMM (Sampled Dense Dense Matrix Multiplication). The code is designed to be run on CPUs and NVIDIA GPUs.

### GPU benchmarks
1. [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) (vendor provided library)
2. Acc-SpMM: [code](https://zenodo.org/records/14214504) [paper](https://dl.acm.org/doi/10.1145/3710848.3710888)
3. ASpT: [code](https://github.com/LucasWilkinson/ASpT-mirror) [paper](https://dl.acm.org/doi/10.1145/3293883.3295712)
4. RoDe: [code](https://github.com/CRAFT-THU/RoDe) [paper](https://dl.acm.org/doi/10.1145/3627535.3638470)
5. HC-SpMM: [code](https://github.com/ZJU-DAILY/HC-SpMM) [paper](https://arxiv.org/abs/2412.08902)
6. dgSPARSE: [code](https://github.com/dgSPARSE/dgSPARSE-Lib) [paper](https://ieeexplore.ieee.org/document/9355302)
7. GNN-Pilot: [code](https://github.com/USTC-ADA/GNNPilot/tree/main) [paper](https://dl.acm.org/doi/10.1145/3730586)
8. DTC-SpMM: [code](https://github.com/HPMLL/DTC-SpMM_ASPLOS24/tree/main) [paper](https://dl.acm.org/doi/abs/10.1145/3620666.3651378)
9. Sputnik: [code](https://github.com/google-research/sputnik) [paper](https://dl.acm.org/doi/10.5555/3433701.3433723)

### CPU benchmarks
1. [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [AMD AOCL-Sparse](https://www.amd.com/en/developer/aocl/sparse.html) (vendor provided libraries)
2. ASpT: [code](https://github.com/LucasWilkinson/ASpT-mirror) [paper](https://dl.acm.org/doi/10.1145/3293883.3295712)
3. FusedMM: [code](https://github.com/HipGraph/FusedMM) [paper](https://ieeexplore.ieee.org/document/9460486)

## Before running the benchmarks

Make sure to download all dependencies (FusedMM, Sputnik, pyTorch with CUDA support):

```bash
git submodule update --init --recursive

cd deps
bash ./install_FusedMM_base.sh
bash ./install_sputnik_base.sh
bash ./install_torch_cuda_base.sh
```

## Running the benchmarks
Edit the Makefile and the run.sh to build the executables needed for the benchmarks.
The paths are configured for the epyc5 server (AMD 24-core CPU and NVIDIA A100 GPU).

In the run.sh file, select the matrices you want to run the benchmarks. You can also configure the number of threads for the CPU programs. 
For every executable, the second input parameter is the dimension k of the dense matrices.

Dimensions of matrices: 
1. Sparse matrix: `m x n`
2. SpMM: input dense matrix: `n x k`, output dense matrix: `m x k`
3. SDDMM: left input dense matrix: `m x k`, right input dense matrix: `k x n`

```bash
make clean
make -j
bash ./run.sh
```

You can edit the `spmm_bench.cpp` and the `sddmm_bench.cpp`, for better reporting of results (no need to edit the `kernel_*` files). In their current form, reported are:
1. the name of the matrix and the number of rows and nonzeros
2. the selected kernel
3. the performance (in GFLOPs) 

Some kernels (dgSPARSE, GNN-Pilot, DTC-SpMM) have several versions of SpMM kernels. After testing with more matrices, we could choose to keep only one (the best performing).

Some kernels (ACC-SpMM, DTC-SpMM) produce wrong results. We choose to ignore it for now...

For the remaining kernels, some matrices with very small values may fail the result verification check, ignore it. This happens due to the selection of 32-bits float numbers.

