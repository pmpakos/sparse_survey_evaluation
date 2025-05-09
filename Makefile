
EXE  = 
# GPU
# EXE += spmm_cusparse.exe sddmm_cusparse.exe 
# EXE += spmm_acc.exe
# EXE += spmm_aspt_gpu.exe sddmm_aspt_gpu.exe
# EXE += spmm_rode.exe sddmm_rode.exe
EXE += spmm_hc.exe

# CPU
# EXE += spmm_mkl.exe
# EXE += spmm_aspt_cpu.exe sddmm_aspt_cpu.exe
# EXE += spmm_aocl.exe
# EXE += spmm_fusedmm.exe

# EXE += mat_dgsparse_spmm.exe mat_dgsparse_sddmm.exe 
# EXE += mat_sputnik_spmm.exe mat_sputnik_sddmm.exe
# EXE += mat_aspt_spmm.exe mat_aspt_sddmm.exe
# EXE += mat_aspt_spmm_cpu.exe mat_aspt_sddmm_cpu.exe
# EXE += mat_aocl_spmv.exe mat_aocl_spmm.exe mat_aocl_spmv3.exe mat_aocl_spmv4.exe 
# EXE += mat_fused_spmm.exe
# EXE += mat_rode_spmm.exe mat_rode_sddmm.exe
# EXE += mat_octet_spmm.exe

# EXE += mat_dtc_v1_spmm.exe mat_dtc_v2_spmm.exe # mat_dtc_v3_spmm.exe
# EXE += mat_hc_spmm.exe
# EXE += mat_gnnpilot_spmm.exe
# EXE += mat_gnnpilot_spmm.exe mat_gnnpilot_sddmm.exe


#####################################################################################################
#####################################################################################################
#####################################################################################################
# COMPILER_PREFIX=/usr/bin
COMPILER_PREFIX=/various/dgal/gcc/gcc-12.2.0/gcc_bin/bin

CC = $(COMPILER_PREFIX)/gcc
# CC = clang

CPP = $(COMPILER_PREFIX)/g++
# CPP = clang++



.PHONY: all clean

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables

# Targets that don't generate dependency files.
NODEPS = clean

DIRS = obj

define Rule_Auto_Dependencies_base =
    $(1:.o=.d): $(2) | $(DIRS)
	@echo 'Generating dependencies file:  $(1)'
	# @echo "$(CC) $(3) -MT '$(1:.o=.d)' -MM -MG '$(2)' -MF '$(1:.o=.d)'"
	$(CC) $(3) -MT '$(1:.o=.d)' -MM -MG '$(2)' -MF '$(1:.o=.d)'
    ifeq (0, $(words $(findstring $(MAKECMDGOALS),$(NODEPS))))
        -include $(1:.o=.d)
    endif
    $(1): $(1:.o=.d)
endef

define Rule_Auto_Dependencies =
    $(eval $(call Rule_Auto_Dependencies_base,$(1),$(2),$(3)))
    $(1): $(2)
endef

CPATH = 

library = ./lib

ARCH = $(shell uname -m)

CFLAGS = -Wall -Wextra
# Tells the compiler to use pipes instead of temporary files (faster compilation, but uses more memory).
CFLAGS += -pipe
CFLAGS += -fopenmp
CFLAGS += -fPIC
ifeq ($(ARCH), x86_64)
    CFLAGS += -mbmi
    CFLAGS += -mbmi2
    CFLAGS += -march=native
	CFLAGS += -mavx2
endif
# CFLAGS += -g3 -fno-omit-frame-pointer
# CFLAGS += -Og
# CFLAGS += -O0
# CFLAGS += -O2
CFLAGS += -O3

# CFLAGS += -ffast-math

# CFLAGS += -flto
# CFLAGS += -march=native

CFLAGS += -D'LEVEL1_DCACHE_LINESIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)'
CFLAGS += -D'LEVEL1_DCACHE_SIZE=$(shell getconf LEVEL1_DCACHE_SIZE)'
CFLAGS += -D'LEVEL2_CACHE_SIZE=$(shell getconf LEVEL2_CACHE_SIZE)'
CFLAGS += -D'LEVEL3_CACHE_SIZE=$(shell getconf LEVEL3_CACHE_SIZE)'

CFLAGS += -I'$(library)'
CFLAGS += -D'_GNU_SOURCE'

CPPFLAGS=''
CPPFLAGS+=" ${CFLAGS}"

#########################

LIBS_ROOT=/various/pmpakos/epyc5_libs
SPARSE_SURVEY_ROOT=$(shell pwd)/deps


########## GPU ##########

CUDA_PATH=/usr/local/cuda
NVCC=${CUDA_PATH}/bin/nvcc -ccbin=${CC}
NVCCFLAGS        = -allow-unsupported-compiler -gencode arch=compute_80,code=sm_80
LDFLAGS_CUSPARSE = -lcuda -lcudart -lcusparse -lcublas

DGSPARSE_PATH=$(SPARSE_SURVEY_ROOT)/dgSPARSE
SPUTNIK_PATH=$(SPARSE_SURVEY_ROOT)/sputnik/build
ACC_PATH=$(SPARSE_SURVEY_ROOT)/ACC-SpMM
ASPT_PATH=$(SPARSE_SURVEY_ROOT)/ASpT
FUSED_PATH=$(SPARSE_SURVEY_ROOT)/FusedMM
RODE_PATH=$(SPARSE_SURVEY_ROOT)/RoDe
OCTET_PATH=$(SPARSE_SURVEY_ROOT)/vectorSparse
DTC_PATH=$(SPARSE_SURVEY_ROOT)/DTC-SpMM
HC_PATH=$(SPARSE_SURVEY_ROOT)/HC-SpMM
GNNPILOT_PATH=$(SPARSE_SURVEY_ROOT)/GNNPilot

TORCH_HOME=$(SPARSE_SURVEY_ROOT)/libtorch
PYTHON_HOME=/various/pmpakos/python-3.9.7

PYTORCH_INC = 
PYTORCH_INC += -I'$(PYTHON_HOME)/include/python3.9'
PYTORCH_INC += -I'$(TORCH_HOME)/include/torch/csrc/api/include/' -I'$(TORCH_HOME)/include'

PYTORCH_LIBS =
PYTORCH_LIBS += -L'$(PYTHON_HOME)/lib' -lpython3.9
PYTORCH_LIBS += -L'$(CUDA_PATH)/lib64' -lcudart -lcublas -lcusparse -lcudnn -lcufft -lcurand
PYTORCH_LIBS += -L'$(TORCH_HOME)/lib'

# Prefer explicit .so linking and handle symbol registration
PYTORCH_LIBS += -Xlinker --no-as-needed -Xlinker $(TORCH_HOME)/lib/libtorch_cpu.so
PYTORCH_LIBS += -Xlinker --no-as-needed -Xlinker $(TORCH_HOME)/lib/libtorch_cuda.so
PYTORCH_LIBS += -Xlinker --no-as-needed -Xlinker $(TORCH_HOME)/lib/libtorch.so
PYTORCH_LIBS += -Xlinker --as-needed

# Remaining essential libs
PYTORCH_LIBS += -lc10 -lc10_cuda


########## CPU ##########

# MKL_PATH = /various/pmpakos/intel/oneapi/mkl/2024.1
MKL_PATH = /various/common_tools/intel_parallel_studio/compilers_and_libraries/linux/mkl
# AOCL_PATH = /various/pmpakos/spmv_paper/aocl-sparse/build/release/
AOCL_ROOT=$(LIBS_ROOT)/aocl-5.0/
# AOCL_PATH=$(AOCL_ROOT)/aocl-sparse/build/release/
AOCL_PATH=$(AOCL_ROOT)/aocl-sparse-dev/build/release/
# AOCL_PATH_3=/various/pmpakos/spmv_paper/aocl-sparse/build/release/
AOCL_PATH3=$(LIBS_ROOT)/aocl-sparse-3.2/build/release/
AOCL_PATH4=$(LIBS_ROOT)/aocl-sparse-4.0/build/release/

CPPFLAGS_MKL = -I'$(MKL_PATH)/include' -I'/usr/include/mkl' -Wno-deprecated-declarations -m64 -mavx2
LDFLAGS_MKL  = -L'$(MKL_PATH)/lib/intel64' -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -ldl

CPPFLAGS_AOCL5 = -I'$(AOCL_PATH)/include' -I'$(AOCL_PATH)/src/include' -I'$(AOCL_PATH)/../../library/src/include' -m64 -mavx2 -std=c++17
CPPFLAGS_AOCL3 = -I'$(AOCL_PATH3)/include' -I'$(AOCL_PATH3)/src/include' -I'$(AOCL_PATH3)/../../library/src/include' -m64 -mavx2
CPPFLAGS_AOCL4 = -I'$(AOCL_PATH4)/include' -I'$(AOCL_PATH4)/src/include' -I'$(AOCL_PATH4)/../../library/src/include' -m64 -mavx2

LDFLAGS_AOCL5 = 
LDFLAGS_AOCL5 += -L'$(LIBS_ROOT)/openssl-1.1.1o/' -L'$(AOCL_ROOT)/amd-blis/lib/LP64/' -L'$(AOCL_ROOT)/amd-libflame/lib/LP64/' -L'$(AOCL_ROOT)/amd-utils/lib/' 
LDFLAGS_AOCL5 += -lflame -lblis-mt -laoclutils
LDFLAGS_AOCL5 += -L'$(AOCL_PATH)/lib/' -Wl,--no-as-needed  -laoclsparse -lgomp -lpthread -ldl

LDFLAGS_AOCL3 = -L'$(AOCL_PATH3)/lib/' -Wl,--no-as-needed  -laoclsparse -lgomp -lpthread -ldl
LDFLAGS_AOCL4 = -L'$(AOCL_PATH4)/lib/' -Wl,--no-as-needed  -laoclsparse -lgomp -lpthread -ldl

#########################


DOUBLE := 0
# double-precision is not supported for most formats! so no point in running experiments with DOUBLE=1...
# DOUBLE := 1

CFLAGS += -D'INT_T=int32_t'

CFLAGS += -D'DOUBLE=$(DOUBLE)'
ifeq ($(DOUBLE), 1)
	CFLAGS += -D'ValueType=double'
	CFLAGS += -D'MATRIX_MARKET_FLOAT_T=double'
else
	CFLAGS += -D'ValueType=float'
	CFLAGS += -D'MATRIX_MARKET_FLOAT_T=float'
endif


LDFLAGS =
LDFLAGS += -lm


LIB_SRC = pthread_functions.c omp_functions.c string_util.c io.c parallel_io.c hash.c random.c array_metrics.c plot.c csr_util.c csr_converter.c matrix_market.c hardware_topology.c
LIB_OBJ := $(LIB_SRC)
LIB_OBJ := $(patsubst %.c,obj/%.o,$(LIB_OBJ))
LIB_OBJ := $(patsubst %.cpp,obj/%.o,$(LIB_OBJ))

all: $(EXE) | $(DIRS)

mat_read.exe: mat_read.cpp $(LIB_OBJ)
	$(CPP) $(CFLAGS) $^ -o $@ $(LDFLAGS)

########## GPU ##########

# mat_cusparse_spmv.exe: mat_cusparse_spmv.cpp $(LIB_OBJ)
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $^ -o $@ $(LDFLAGS) $(LDFLAGS_CUSPARSE)
spmm_cusparse.exe: obj/spmm_bench.o kernel_cusparse.cu $(LIB_OBJ)
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $^ -o $@ $(LDFLAGS) $(LDFLAGS_CUSPARSE)
sddmm_cusparse.exe: obj/sddmm_bench.o kernel_cusparse.cu $(LIB_OBJ)
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $^ -o $@ $(LDFLAGS) $(LDFLAGS_CUSPARSE)

spmm_acc.exe: obj/spmm_bench.o kernel_acc.cu $(LIB_OBJ)
	cd $(ACC_PATH); make clean; make -j; cd -
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -I'$(ACC_PATH)'" $^ -o $@ $(LDFLAGS) -L'$(ACC_PATH)' -lacc_spmm

spmm_aspt_gpu.exe: obj/spmm_bench.o kernel_aspt_gpu.cu $(LIB_OBJ)
	cd $(ASPT_PATH)/gpu/spmm; make clean; make DOUBLE=$(DOUBLE) -j; cd -
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -D'SPMM_KERNEL' -I'$(ASPT_PATH)/gpu'" $^ -o $@ $(LDFLAGS) -L'$(ASPT_PATH)/gpu/spmm' -laspt_spmm
sddmm_aspt_gpu.exe: obj/sddmm_bench.o kernel_aspt_gpu.cu $(LIB_OBJ)
	cd $(ASPT_PATH)/gpu/sddmm/; make clean; make DOUBLE=$(DOUBLE) -j; cd -
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -D'SDDMM_KERNEL' -I'$(ASPT_PATH)/gpu'" $^ -o $@ $(LDFLAGS) -L'$(ASPT_PATH)/gpu/sddmm/' -laspt_sddmm

spmm_rode.exe: obj/spmm_bench.o kernel_rode.cu $(LIB_OBJ)
	cd $(RODE_PATH)/spmm; make clean; make -j; cd -
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -D'SPMM_KERNEL' -I'$(RODE_PATH)'" $^ -o $@ $(LDFLAGS) -L'$(RODE_PATH)/spmm' -lrode_spmm
sddmm_rode.exe: obj/sddmm_bench.o kernel_rode.cu $(LIB_OBJ)
	cd $(RODE_PATH)/sddmm; make clean; make -j; cd -
	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -D'SDDMM_KERNEL' -I'$(RODE_PATH)'" $^ -o $@ $(LDFLAGS) -L'$(RODE_PATH)/sddmm' -lrode_sddmm

spmm_hc.exe: obj/spmm_bench.o kernel_hc.cu $(LIB_OBJ)
# 	cd $(HC_PATH); make clean; make TORCH_HOME=$(TORCH_HOME) PYTHON_HOME=$(PYTHON_HOME) -j; cd -
	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(HC_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(HC_PATH)' -lhc_spmm $(PYTORCH_LIBS)

########## CPU ##########

# mat_mkl_spmv.exe: mat_mkl_spmv.cpp $(LIB_OBJ)
# 	$(CPP) $(CFLAGS) $(CPPFLAGS_MKL) $^ -o $@ $(LDFLAGS) $(LDFLAGS_MKL)
spmm_mkl.exe: obj/spmm_bench.o kernel_mkl.cpp $(LIB_OBJ)
	$(CPP) $(CFLAGS) $(CPPFLAGS_MKL) $^ -o $@ $(LDFLAGS) $(LDFLAGS_MKL)

spmm_aspt_cpu.exe: obj/spmm_bench.o kernel_aspt_cpu.cpp $(LIB_OBJ)
	cd $(ASPT_PATH)/cpu/spmm/; make clean; make DOUBLE=$(DOUBLE) -j; cd -
	$(CPP) $(CFLAGS) -D'SPMM_KERNEL' $^ -o $@ -I'$(ASPT_PATH)/cpu' $(LDFLAGS) -L'$(ASPT_PATH)/cpu/spmm' -laspt_spmm
sddmm_aspt_cpu.exe: obj/sddmm_bench.o kernel_aspt_cpu.cpp $(LIB_OBJ)
	cd $(ASPT_PATH)/cpu/sddmm/; make clean; make DOUBLE=$(DOUBLE) -j; cd -
	$(CPP) $(CFLAGS) -D'SDDMM_KERNEL' $^ -o $@ -I'$(ASPT_PATH)/cpu' $(LDFLAGS) -L'$(ASPT_PATH)/cpu/sddmm' -laspt_sddmm

# mat_aocl_spmv.exe: mat_aocl_spmv.cpp $(LIB_OBJ)
# 	$(CPP) $(CFLAGS) $(CPPFLAGS_AOCL5) $^ -o $@ $(LDFLAGS) $(LDFLAGS_AOCL5)
# mat_aocl_spmv3.exe: mat_aocl_spmv3.cpp $(LIB_OBJ)
# 	$(CPP) $(CFLAGS) $(CPPFLAGS_AOCL3) $^ -o $@ $(LDFLAGS) $(LDFLAGS_AOCL3)
# mat_aocl_spmv4.exe: mat_aocl_spmv4.cpp $(LIB_OBJ)
# 	$(CPP) $(CFLAGS) $(CPPFLAGS_AOCL4) $^ -o $@ $(LDFLAGS) $(LDFLAGS_AOCL4)
spmm_aocl.exe: obj/spmm_bench.o kernel_aocl.cpp $(LIB_OBJ)
	$(CPP) $(CFLAGS) $(CPPFLAGS_AOCL5) $^ -o $@ $(LDFLAGS) $(LDFLAGS_AOCL5)

# mat_fused_sddmm.exe: mat_fused_sddmm.cpp $(LIB_OBJ)
# 	cd $(FUSED_PATH); make clean; make killlib; make -j > /dev/null; cd -;
# 	$(CPP) $(CFLAGS) $^ -o $@ -I'$(FUSED_PATH)' $(LDFLAGS) $(FUSED_PATH)/bin/sOptFusedMM_pt.o $(FUSED_PATH)/kernels/lib/slibgfusedMM_pt.a
spmm_fusedmm.exe: obj/spmm_bench.o kernel_fusedmm.cpp $(LIB_OBJ)
	cd $(FUSED_PATH); make clean; make killlib; make -j > /dev/null 2>&1; cd -;
	$(CPP) $(CFLAGS) $^ -o $@ -I'$(FUSED_PATH)' $(LDFLAGS) $(FUSED_PATH)/bin/sOptFusedMM_pt.o $(FUSED_PATH)/kernels/lib/slibgfusedMM_pt.a


# mat_ge_spmm.exe: mat_ge_spmm.cu $(LIB_OBJ)
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $^ -o $@ $(LDFLAGS) $(LDFLAGS_CUSPARSE)
# mat_dgsparse_spmm.exe: mat_dgsparse_spmm.cpp $(LIB_OBJ)
# 	cd $(DGSPARSE_PATH)/ge-spmm; make clean; make -j; cd -
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -I'$(DGSPARSE_PATH)/ge-spmm'" $^ -o $@ $(LDFLAGS) -L'$(DGSPARSE_PATH)/ge-spmm' -lgespmm
# mat_dgsparse_sddmm.exe: mat_dgsparse_sddmm.cpp $(LIB_OBJ)
# 	cd $(DGSPARSE_PATH)/sddmm; make clean; make -j; cd -
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -I'$(DGSPARSE_PATH)/sddmm'" $^ -o $@ $(LDFLAGS) -L'$(DGSPARSE_PATH)/sddmm' -lsddmm

# mat_sputnik_spmm.exe: mat_sputnik_spmm.cpp $(LIB_OBJ)
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -I'$(SPUTNIK_PATH)/include'" $^ -o $@ $(LDFLAGS) -L'$(SPUTNIK_PATH)/lib' -lsputnik
# mat_sputnik_sddmm.exe: mat_sputnik_sddmm.cpp $(LIB_OBJ)
# 	$(NVCC) $(NVCCFLAGS) --compiler-options "$(CFLAGS) -I'$(SPUTNIK_PATH)/include'" $^ -o $@ $(LDFLAGS) -L'$(SPUTNIK_PATH)/lib' -lsputnik

# mat_dtc_v1_spmm.exe: mat_dtc_v1_spmm.cu $(LIB_OBJ)
# 	cd $(DTC_PATH); make clean; make -j; cd -
# 	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(DTC_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(DTC_PATH)' -ldtc_spmm $(PYTORCH_LIBS)

# mat_dtc_v2_spmm.exe: mat_dtc_v2_spmm.cu $(LIB_OBJ)
# 	cd $(DTC_PATH); make clean; make -j; cd -
# 	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(DTC_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(DTC_PATH)' -ldtc_spmm $(PYTORCH_LIBS) 

# # mat_dtc_v3_spmm.exe: mat_dtc_v3_spmm.cu $(LIB_OBJ)
# # 	cd $(DTC_PATH); make clean; make -j; cd -
# # 	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(DTC_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(DTC_PATH)' -ldtc_spmm $(PYTORCH_LIBS) 

# mat_gnnpilot_spmm.exe: mat_gnnpilot_spmm.cu $(LIB_OBJ)
# 	cd $(GNNPILOT_PATH); make clean; make -j; cd -
# 	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(GNNPILOT_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(GNNPILOT_PATH)' -lgnnpilot $(PYTORCH_LIBS)
# mat_gnnpilot_sddmm.exe: mat_gnnpilot_sddmm.cu $(LIB_OBJ)
# 	cd $(GNNPILOT_PATH); make clean; make -j; cd -
# 	$(NVCC) -std=c++17 $(NVCCFLAGS) --compiler-options "-std=c++17 $(CFLAGS) -I'$(GNNPILOT_PATH)' $(PYTORCH_INC)" $^ -o $@ $(LDFLAGS) -L'$(GNNPILOT_PATH)' -lgnnpilot $(PYTORCH_LIBS)


# ########## CPU ##########

# #########################

$(call Rule_Auto_Dependencies,obj/spmm_bench.o,spmm_bench.cpp,$(CFLAGS))
	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/sddmm_bench.o,sddmm_bench.cpp,$(CFLAGS))
	$(CPP) $(CFLAGS) -c $< -o $@

# $(call Rule_Auto_Dependencies,obj/read_mtx.o,read_mtx.cpp,$(CFLAGS))
# 	$(CPP) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/pthread_functions.o,$(library)/pthread_functions.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/omp_functions.o,$(library)/omp_functions.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/string_util.o,$(library)/string_util.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/io.o,$(library)/io.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/parallel_io.o,$(library)/parallel_io.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/hash.o,$(library)/hash/hash.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/random.o,$(library)/random.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/array_metrics.o,$(library)/array_metrics.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/plot.o,$(library)/plot/plot.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr_converter.o,$(library)/aux/csr_converter.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/csr_util.o,$(library)/aux/csr_util.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/matrix_market.o,$(library)/storage_formats/matrix_market/matrix_market.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@
$(call Rule_Auto_Dependencies,obj/hardware_topology.o,$(library)/topology/hardware_topology.c,$(CFLAGS))
	$(CC) $(CFLAGS) -c $< -o $@

$(DIRS): %:
	mkdir -p $@

clean:
	$(RM) obj/*.o obj/*.d *.o *.exe
