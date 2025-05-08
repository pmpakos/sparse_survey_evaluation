#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <cuda_runtime.h>
#include <string>
#include <cstring>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line);
void checkLast(const char* const file, const int line);

void printCudaInfo();

std::string match_filename(std::string s);

void init_vec(const vint rows, const vint cols, MAT_VAL_TYPE** Mat);

void init_vec1(const vint nnz, MAT_VAL_TYPE* Mat, MAT_VAL_TYPE val);

void init_vecB(const vint rows, const vint cols, MAT_VAL_TYPE* Mat, MAT_VAL_TYPE val);

void freeDense(const vint rows, MAT_VAL_TYPE** Mat);

struct GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;
	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() { cudaEventRecord(start); }

	void Stop()  { cudaEventRecord(stop); }

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

typedef uint64_t clocktype;
struct Dur {
	clocktype begin;
	clocktype end;
	int smid = -1;
	Dur(clocktype x, clocktype y, int outsm) {
		begin = x;
		end = y;
		smid = outsm;
	}
};

// __device__
// void acquireLock(int *lock) {
//     while (atomicCAS(lock, 0, 1) != 0) {
//         // Busy-wait until the lock is acquired
//     }
// }

// __device__
// void releaseLock(int *lock) {
//     atomicExch(lock, 0);
// }

// __device__ int lock = 0;

typedef struct {
	float f1, f2, f3, f4, f5, f6, f7, f8;
} float8;

#endif // UTILS_H
