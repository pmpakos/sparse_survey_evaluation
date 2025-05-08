#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <string>
#include <cstring>
#include "class.h"

typedef char MM_typecode[4];
int mm_read_banner(FILE* file, MM_typecode* matcode);
int mm_read_mtx_crd_size(FILE* file, vint* M, vint* N, vint* nz);
template<class dataType>
int read_from_mtx(char* filename, COO<dataType>* coo);

template<class dataType>
void handle_coo(FILE* file, vint m, vint n, vint nnz, bool isInteger, bool isReal, bool isPattern, bool isSymmetric, bool isComplex, COO<dataType>* coo);
bool mm_is_valid(MM_typecode matcode);		/* too complex for a macro */

/* ------------------------ */
#endif // LOAD_DATA_H