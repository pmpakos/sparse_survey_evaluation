#compilers

# COMPILER_PREFIX=/usr/bin
COMPILER_PREFIX=/various/dgal/gcc/gcc-12.2.0/gcc_bin/bin

CC = $(COMPILER_PREFIX)/g++

TARGET = libaspt_spmm.a

SRCS = $(wildcard *.cc)
OBJS = $(patsubst %cc, %cc.o, $(SRCS))

C_FLAGS = 

C_FLAGS += -Wall -Wextra
# Tells the compiler to use pipes instead of temporary files (faster compilation, but uses more memory).
C_FLAGS += -pipe
C_FLAGS += -fopenmp

ARCH = $(shell uname -m)
ifeq ($(ARCH), x86_64)
    C_FLAGS += -mbmi
    C_FLAGS += -mbmi2
    C_FLAGS += -march=native
	C_FLAGS += -mavx2
endif
C_FLAGS += -O3
C_FLAGS += -ffast-math -funroll-loops
C_FLAGS += -DDOUBLE=$(DOUBLE)

INC = -I ./nlibs

all: $(TARGET)

$(TARGET) : $(OBJS)
	ar cr $@ *.o
	ranlib $@

%.cc.o: %.cc
	$(CXX) $(C_FLAGS) $(INC) -c $< -o $@

clean:
	rm -rf $(TARGET)
	rm -rf *.o
