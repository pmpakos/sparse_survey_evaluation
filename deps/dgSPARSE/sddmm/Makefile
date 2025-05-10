TARGET = libsddmm.a

SRCS = $(wildcard *.cu)
OBJS = $(patsubst %cu, %cu.o, $(SRCS))

CUDA_HOME=/usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

NVCC_FLAGS = 
NVCC_FLAGS += -O3 -lcudart
NVCC_FLAGS += -shared -std=c++17
NVCC_FLAGS += -Xcompiler -fPIC
NVCC_FLAGS += -Xcompiler -fopenmp
NVCC_FLAGS += -ccbin /usr/bin
NVCC_FLAGS += -m64
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80
NVCC_FLAGS += -diag-suppress=191,177

INC = -I$(CUDA_HOME)/include

all : $(TARGET)

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@

%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
