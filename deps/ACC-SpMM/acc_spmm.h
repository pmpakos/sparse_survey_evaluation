#ifndef ACC_SPMM_H
#define ACC_SPMM_H

#include "common.h"
#include "utils.h"
#include "class.h"
void tf32_spmm(
	METCFBit<MAT_VAL_TYPE>& metcf_bit, 
	BME<MAT_VAL_TYPE>& bme, 
	AdpBME<MAT_VAL_TYPE>& adpbme,
	COO<MAT_VAL_TYPE>* coo,
	const vint feature_dim, 
	const std::string filename, 
	bool load_balance
);

#endif // ACC_SPMM_H
