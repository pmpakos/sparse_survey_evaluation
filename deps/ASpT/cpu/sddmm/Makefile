TARGET = libaspt_sddmm.a

SRCS = $(wildcard *.cc)
OBJS = $(patsubst %cc, %cc.o, $(SRCS))

C_FLAGS = -Wall -Wextra -pipe -fopenmp
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

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@

%.cc.o: %.cc
	$(CPP) $(C_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
